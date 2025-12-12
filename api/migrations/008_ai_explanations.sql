-- Create table for tracking AI usage
CREATE TABLE IF NOT EXISTS ai_explanations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id),
    job_id UUID REFERENCES jobs(id),
    question TEXT,
    response_length INT,
    cost_estimate DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add RLS policies
ALTER TABLE ai_explanations ENABLE ROW LEVEL SECURITY;

-- Users can view their own explanations
CREATE POLICY "Users can view own explanations" ON ai_explanations
    FOR SELECT USING (auth.uid() = user_id);

-- Only service role can insert (for now, or users if we want client-side tracking but backend is better)
-- Actually, backend inserts it, so service role or postgres role.
-- But if we use supabase-py in backend with service key, it bypasses RLS.
-- If we use authenticated client, we need insert policy.
CREATE POLICY "Users can insert own explanations" ON ai_explanations
    FOR INSERT WITH CHECK (auth.uid() = user_id);
