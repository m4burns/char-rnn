
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'

require 'util.OneHot'
require 'util.misc'
local NoteMinibatchLoader = require 'util.NoteMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

gnuplot.setterm('x11')
gnuplot.figure()
gnuplot.axis({0.3,0.6,0.3,0.6})

cmd = torch.CmdLine()
-- data
cmd:option('-data_dir','data/song','data directory')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',1,'number of sequences to train on in parallel') -- was 50
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = NoteMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(2, opt.rnn_size, opt.num_layers, opt.dropout)
    --[[
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    --]]
    end
    protos.criterion = nn.MSECriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
params:uniform(-0.20, 0.20) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state_ctx = {[0] = init_state}
    local rnn_state_add = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x = loader:next_batch(split_index)

        local label = x.label
        local context = x.context:double()
        local addition = x.addition:double()

        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            context = context:cuda()
            addition = addition:cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            context = context:cl()
            addition = addition:cl()
        end

        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning

            -- on context
            local lst = clones.rnn[t]:forward{context[t]:resize(1,2), unpack(rnn_state_ctx[t-1])}
            rnn_state_ctx[t] = {}
            for i=1,#init_state do table.insert(rnn_state_ctx[t], lst[i]) end

            local prediction_ctx = lst[#lst]:clone()

            -- on addition
            lst = clones.rnn[t]:forward{addition[t]:resize(1,2), unpack(rnn_state_add[t-1])}
            rnn_state_add[t] = {}
            for i=1,#init_state do table.insert(rnn_state_add[t], lst[i]) end

            local prediction_add = lst[#lst]:clone()

            if label == "good" then
              loss = loss + clones.criterion[t]:forward(prediction_ctx, prediction_add)
            else
              -- XXX pick a more reasoned loss function
              -- loss = loss + 1 / clones.criterion[t]:forward(prediction_ctx, prediction_add)
              -- loss = loss + clones.criterion[t]:forward(prediction_ctx, prediction_add)
            end
        end
        -- carry over lstm state
        rnn_state_add[0] = rnn_state_add[#rnn_state_add]
        rnn_state_ctx[0] = rnn_state_ctx[#rnn_state_ctx]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

function make_bad_dir(ctx, add)
  local add_to_ctx = ctx:clone():add(add, -1)
  local len = add_to_ctx:norm()
  --if len == 0 then
    return torch.rand(1,2)
  --end
  --else
  --  return add:clone():add(add_to_ctx, -1 / len)
  --end
end

-- local foo = torch.rand(1,20)

local last_label = ""
local last_distance = 0

local pv_add = torch.Tensor(100,2)
local pv_ctx = torch.Tensor(100,2)
local pv_ctr = 1

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x = loader:next_batch(1)
    local context = x.context:double()
    local addition = x.addition:double()
    local label = x.label

    last_label = label

    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        context = context:cuda()
        addition = addition:cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        context = context:cl()
        addition = addition:cl()
    end
    ------------------- forward pass on context and addition -------------------
    local rnn_state_ctx = {[0] = init_state_global}
    local rnn_state_add = {[0] = init_state_global}
    local predictions_ctx = {}           -- softmax outputs
    local predictions_add = {}           -- softmax outputs
    local loss = 0
    local total_dist = 0
    local bad_dirs = {}

    for t=1,opt.seq_length do
        clones.rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)

        context_tr = context[t]:resize(1,2)
        local lst = clones.rnn[t]:forward{context_tr, unpack(rnn_state_ctx[t-1])}
        rnn_state_ctx[t] = {}
        for i=1,#init_state do table.insert(rnn_state_ctx[t], lst[i]) end -- extract the state, without output
        predictions_ctx[t] = lst[#lst]:clone() -- last element is the prediction
    end

    grad_params:zero()

    for t=1,opt.seq_length do
        clones.rnn[t]:training()

        addition_tr = addition[t]:resize(1,2)
        local lst = clones.rnn[t]:forward{addition_tr, unpack(rnn_state_add[t-1])}
        rnn_state_add[t] = {}
        for i=1,#init_state do table.insert(rnn_state_add[t], lst[i]) end -- extract the state, without output
        predictions_add[t] = lst[#lst]:clone() -- last element is the prediction

        if label == "good" then
          local dist = clones.criterion[t]:forward(predictions_add[t], predictions_ctx[t])
          loss = loss + dist
        else
          bad_dirs[t] = make_bad_dir(predictions_ctx[t], predictions_add[t])
          loss = loss + clones.criterion[t]:forward(predictions_add[t], bad_dirs[t])
        end

        total_dist = total_dist + torch.dist(predictions_ctx[t], predictions_add[t])
    end
    loss = loss / opt.seq_length
    last_distance = total_dist / opt.seq_length

    pv_add[pv_ctr] = predictions_add[opt.seq_length]
    pv_ctx[pv_ctr] = predictions_ctx[opt.seq_length]
    pv_ctr = pv_ctr + 1
    if pv_ctr > 100 then
      pv_ctr = 1
    end

    gnuplot.plot({'Addition',pv_add,'+'}, {'Context',pv_ctx,'+'})
    --gnuplot.plotflush()

    ------------------ backward pass on addition -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t
        if label == "good" then
          doutput_t = clones.criterion[t]:backward(predictions_add[t], predictions_ctx[t])
        else
          doutput_t = clones.criterion[t]:backward(predictions_add[t], bad_dirs[t])
        end
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({addition[t]:resize(1,2), unpack(rnn_state_add[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state_add[#rnn_state_add] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, label = %s, distance = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, last_label, last_distance, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    --[[if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end]]--
end


