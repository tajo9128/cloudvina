-- Add admin flag to existing profiles table
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;

-- Create admin_actions audit table
CREATE TABLE IF NOT EXISTS admin_actions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    admin_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    target_id TEXT,
    target_type TEXT,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create job_queue_control table
CREATE TABLE IF NOT EXISTS job_queue_control (
    id SERIAL PRIMARY KEY,
    max_concurrent_jobs INTEGER DEFAULT 10,
    max_jobs_per_hour INTEGER DEFAULT 100,
    maintenance_mode BOOLEAN DEFAULT FALSE,
    maintenance_message TEXT,
    updated_by UUID REFERENCES auth.users(id),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default control settings if not exists
INSERT INTO job_queue_control (max_concurrent_jobs, max_jobs_per_hour) 
SELECT 10, 100
WHERE NOT EXISTS (SELECT 1 FROM job_queue_control);

-- Enable RLS
ALTER TABLE admin_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_queue_control ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for admin tables
-- Note: We use a subquery to check admin status because we can't join in RLS policies easily without infinite recursion risks sometimes, 
-- but here we check the profiles table which is separate.

DROP POLICY IF EXISTS "Only admins can view admin_actions" ON admin_actions;
CREATE POLICY "Only admins can view admin_actions" ON admin_actions
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.is_admin = TRUE
        )
    );

DROP POLICY IF EXISTS "Only admins can insert admin_actions" ON admin_actions;
CREATE POLICY "Only admins can insert admin_actions" ON admin_actions
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.is_admin = TRUE
        )
    );

DROP POLICY IF EXISTS "Only admins can modify queue control" ON job_queue_control;
CREATE POLICY "Only admins can modify queue control" ON job_queue_control
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.is_admin = TRUE
        )
    );

-- Allow read access to queue control for everyone (so the app knows if maintenance mode is on)
DROP POLICY IF EXISTS "Everyone can read queue control" ON job_queue_control;
CREATE POLICY "Everyone can read queue control" ON job_queue_control
    FOR SELECT USING (TRUE);
