-- Migration 005: Add Docking Results Storage
-- Adds columns to store parsed AutoDock Vina results

-- Add new columns for summary statistics
ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS best_affinity FLOAT,
ADD COLUMN IF NOT EXISTS num_poses INTEGER,
ADD COLUMN IF NOT EXISTS energy_range_min FLOAT,
ADD COLUMN IF NOT EXISTS energy_range_max FLOAT,
ADD COLUMN IF NOT EXISTS docking_results JSONB;

-- Add index for faster queries on affinity
CREATE INDEX IF NOT EXISTS idx_jobs_best_affinity ON jobs(best_affinity);

-- Add comment for documentation
COMMENT ON COLUMN jobs.best_affinity IS 'Best (most negative) binding affinity in kcal/mol';
COMMENT ON COLUMN jobs.num_poses IS 'Number of docking poses generated';
COMMENT ON COLUMN jobs.docking_results IS 'Full parsed results from Vina output (poses, RMSD, etc.)';

-- Grant permissions
GRANT SELECT ON jobs TO authenticated;
