-- Add receptor_filename and ligand_filename columns to jobs table
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS receptor_filename TEXT,
ADD COLUMN IF NOT EXISTS ligand_filename TEXT;
