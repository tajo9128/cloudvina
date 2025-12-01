-- Migration: Add filename columns to jobs table
-- Created: 2025-12-01
-- Purpose: Store original filenames for display on job results page

-- Add columns to store original filenames
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS receptor_filename TEXT,
ADD COLUMN IF NOT EXISTS ligand_filename TEXT;

-- Add comment for documentation
COMMENT ON COLUMN jobs.receptor_filename IS 'Original filename of the uploaded receptor file';
COMMENT ON COLUMN jobs.ligand_filename IS 'Original filename of the uploaded ligand file';
