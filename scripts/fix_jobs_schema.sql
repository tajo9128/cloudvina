-- Fix missing columns in jobs table
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
