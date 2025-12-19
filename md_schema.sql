-- Isolated Table for MD Stability Analysis
-- Prefix: md_ (as requested)

CREATE TABLE IF NOT EXISTS md_stability_jobs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL, -- References auth.users(id) usually, or public.users
    molecule_name TEXT,
    rmsd FLOAT NOT NULL,
    rmsf FLOAT NOT NULL,
    md_score FLOAT, -- The AI Prediction (0-100)
    bucket_used TEXT, -- To track the isolated bucket 'biodockify-md-stability-engine'
    status TEXT DEFAULT 'PENDING', -- 'PENDING', 'SUCCESS', 'FAILED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Optional: Link to a Docking Job if this MD was derived from one
    docking_job_id UUID 
);

-- Row Level Security (RLS) Policies (Standard Supabase Isolation)
ALTER TABLE md_stability_jobs ENABLE ROW LEVEL SECURITY;

-- Allow users to see only their own jobs
CREATE POLICY "Users can view own MD jobs" 
ON md_stability_jobs FOR SELECT 
USING (auth.uid() = user_id);

-- Allow users to insert their own jobs
CREATE POLICY "Users can insert own MD jobs" 
ON md_stability_jobs FOR INSERT 
WITH CHECK (auth.uid() = user_id);
