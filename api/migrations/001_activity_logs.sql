-- Phase 6: Feature Mega-Update - Database Migrations

-- Create activity_logs table for audit trail
CREATE TABLE IF NOT EXISTS activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_activity_logs_user ON activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created ON activity_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_action ON activity_logs(action);
CREATE INDEX IF NOT EXISTS idx_activity_logs_resource ON activity_logs(resource_type, resource_id);

-- Add parameters column to jobs table for advanced docking parameters
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS parameters JSONB DEFAULT '{}';

-- Add comment for documentation
COMMENT ON TABLE activity_logs IS 'Audit trail for all significant system actions';
COMMENT ON COLUMN jobs.parameters IS 'Advanced docking parameters (exhaustiveness, num_modes, energy_range)';
