require 'lfs'
local json = require 'json'

local NoteMinibatchLoader = {}
NoteMinibatchLoader.__index = NoteMinibatchLoader

function NoteMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, NoteMinibatchLoader)

    local tensor_file = path.join(data_dir, 'data.t7')

    print('loading data...')
    local data = torch.load(tensor_file)

    -- cut off the end so that it divides evenly
    local len = #data

    local toremove = len % batch_size

    if toremove > 0 then
      print('cutting off end of data so that the batches/sequences divide evenly')
      for i = 1,toremove do
        table.remove(data, #data)
      end
    end

    len = #data

    print('shuffling data...')
    for i = len, 2, -1 do
      local j = math.random(len)
      data[i], data[j] = data[j], data[i]
    end

    -- self.batches is a table of { context=, addition=, label= }
    print('reshaping tensors...')
    function reshape(tens)
      local orig_rows = tens:size(1)
      tens:resize(seq_length, tens:size(2))
      if(orig_rows < seq_length) then
        tens[{{orig_rows,seq_length}}]:zero()
      end
      return tens
    end
    for i = 1, len do
      data[i].context = reshape(data[i].context)
      data[i].addition = reshape(data[i].addition)
    end

    self.batch_size = batch_size
    self.seq_length = seq_length

    -- XXX for now just do batch_size = 1 to simplify this class
    -- self.batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.batches = data
    self.nbatches = #self.batches

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function NoteMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function NoteMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.batches[ix]
end

-- *** STATIC method ***
function NoteMinibatchLoader.text_to_tensor(in_dir, data_dir)
    local timer = torch.Timer()

    local res = {}

    for _, label in pairs({ 'good', 'bad' }) do
      for file in lfs.dir(path.join(in_dir, label)) do
        file = path.join(in_dir, label, file)
        if lfs.attributes(file).mode == 'file' then
          print(file)

          local f = torch.DiskFile(file, "r")
          local js = json.decode(f:readString("*a"))
          f:close()

          local context = NoteMinibatchLoader.sample_notelist(js.context)
          local addition = NoteMinibatchLoader.sample_notelist(js.addition)

          if context ~= nil and addition ~= nil then
            res[#res + 1] = {
              label = label,
              context  = context,
              addition = addition
            }
          end

          if #res % 10 == 0 then
            collectgarbage()
          end

          --[[
          if #res > 100 then
            for _, t in pairs(res) do
              print(t.label)
              print(t.context)
              print(t.addition)
            end
            torch.save(path.join(data_dir, 'data.t7'), res)
            os.exit(0)
          end
          ]]--
        end
      end
    end

    torch.save(path.join(data_dir, 'data.t7'), res)
end

function NoteMinibatchLoader.sample_notelist(notes)
    -- count the events in the first bar
    local evt_count = 0

    table.sort(notes, function(a, b)
        return a.start < b.start
      end)

    for _, evt in pairs(notes) do
      if evt.pitch ~= nil then
        if evt.start < (4 * 960) then
          evt_count = evt_count + 1
        else
          break
        end
      end
    end

    if evt_count < 2 then
      return nil
    end

    -- keep a map of what pitches are sounding
    local pitches = {}

    -- construct a tensor with all the data
    local data = torch.ByteTensor(evt_count, 20)

    local i = 1, j

    for _, evt in pairs(notes) do
      if evt.start >= (4 * 960) then
        break
      end

      if evt.pitch ~= nil then
        local extent = evt.start + evt.length
        local p = evt.pitch % 12
        if pitches[p] ~= nil then
          pitches[p] = math.max(pitches[p], extent)
        else
          pitches[p] = extent
        end

        local ts = math.floor(evt.start * (255 / (960 * 4)))
  
        for j = 1, 8 do
          local sub = 2^(8-j)
          if sub < ts then
            ts = ts - sub
            data[i][j] = 1
          else
            data[i][j] = 0
          end
        end
  
        for j = 9, 20 do
          local q = j - 9
          if pitches[q] ~= nil and pitches[q] > evt.start then
            if q == p then
              data[i][j] = 2
            else
              data[i][j] = 1
            end
          else
            data[i][j] = 0
          end
        end
      end

      i = i + 1
    end

    return data
end

return NoteMinibatchLoader

