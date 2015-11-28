local NoteMinibatchLoader = require 'util.NoteMinibatchLoader'

-- NoteMinibatchLoader.text_to_tensor('data/song', 'data/song')

local nmbl = NoteMinibatchLoader.create('data', 8, 50, { 0.9, 0.05, 0.05 })
