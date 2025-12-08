-- User System Upgrade: Support, Profile, Billing
-- Run this in Supabase SQL Editor

-- 1. Remove Rate Limiting (Drop Table & Function)
DROP TABLE IF EXISTS daily_job_usage CASCADE;
DROP FUNCTION IF EXISTS get_daily_limit(UUID);

-- 2. Enhance User Profiles
-- Ensure phone, designation, organization columns exist (should be there from 006)
-- Add social_links and verification flags
ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS social_links JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS phone_verified BOOLEAN DEFAULT FALSE;

-- 3. Create Support Tickets System
CREATE TABLE IF NOT EXISTS support_tickets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    category TEXT NOT NULL, -- e.g., 'Billing', 'Technical', 'Account'
    message TEXT NOT NULL,
    status TEXT DEFAULT 'open', -- 'open', 'in_progress', 'resolved', 'closed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on Support Tickets
ALTER TABLE support_tickets ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view their own tickets
CREATE POLICY "Users can view own tickets" ON support_tickets
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can create tickets
CREATE POLICY "Users can create tickets" ON support_tickets
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Admins can view all tickets (using user_credits or profiles is_admin check)
CREATE POLICY "Admins can view all tickets" ON support_tickets
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM profiles 
            WHERE profiles.id = auth.uid() AND profiles.is_admin = TRUE
        )
    );

-- Policy: Admins can update tickets (reply/close)
CREATE POLICY "Admins can update tickets" ON support_tickets
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM profiles 
            WHERE profiles.id = auth.uid() AND profiles.is_admin = TRUE
        )
    );

-- 4. Backfill Free Plan
UPDATE user_credits 
SET plan = 'free' 
WHERE plan IS NULL;

-- 5. Create Trigger for Updated At
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_support_tickets_updated_at
    BEFORE UPDATE ON support_tickets
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();
