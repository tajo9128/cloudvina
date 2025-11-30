-- Migration 006: Add Interaction Results Storage
-- Adds column to store parsed protein-ligand interactions

ALTER TABLE jobs 
ADD COLUMN IF NOT EXISTS interaction_results JSONB;

COMMENT ON COLUMN jobs.interaction_results IS 'Parsed protein-ligand interactions (H-bonds, hydrophobic contacts)';

-- Grant permissions
GRANT SELECT ON jobs TO authenticated;
