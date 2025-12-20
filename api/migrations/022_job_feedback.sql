-- Migration: 022_job_feedback
-- Description: Create job_feedback table for Active Learning loop

CREATE TABLE IF NOT EXISTS job_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)), -- 1 = Like/Good, -1 = Dislike/Bad
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Constraint: User can only rate a job once (or we can allow update, let's allow unique per job-user)
    UNIQUE(job_id, user_id)
);

-- Enable RLS
ALTER TABLE job_feedback ENABLE ROW LEVEL SECURITY;

-- Policies

-- 1. Users can insert their own feedback
CREATE POLICY "Users can insert own feedback" ON job_feedback
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 2. Users can view their own feedback
CREATE POLICY "Users can view own feedback" ON job_feedback
    FOR SELECT USING (auth.uid() = user_id);

-- 3. Users can update their own feedback
CREATE POLICY "Users can update own feedback" ON job_feedback
    FOR UPDATE USING (auth.uid() = user_id);

-- 4. Admins can view all feedback (for training)
CREATE POLICY "Admins can view all feedback" ON job_feedback
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM rbac_user_roles ur
            JOIN rbac_roles r ON ur.role_id = r.id
            WHERE ur.user_id = auth.uid() AND r.name IN ('admin', 'manager')
        )
    );
