-- Fix RLS Policy for Signup
-- Run this in Supabase SQL Editor

-- Enable RLS on user_credits if not already enabled (good practice)
ALTER TABLE user_credits ENABLE ROW LEVEL SECURITY;

-- 1. Allow Service Role (Backend) full access
-- This is critical for the API to insert/update credits
CREATE POLICY "Service Role Full Access" ON user_credits
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 2. Allow Users to View their own credits
CREATE POLICY "Users can view own credits" ON user_credits
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

-- 3. Allow Users to Insert their own credits (for signup)
-- This is likely the missing piece causing the 42501 error
CREATE POLICY "Users can insert own credits" ON user_credits
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- 4. Allow Users to Update their own credits (optional, usually backend handles this)
-- But safe to allow if restricted to own user_id
-- CREATE POLICY "Users can update own credits" ON user_credits
--     FOR UPDATE
--     TO authenticated
--     USING (auth.uid() = user_id);

-- Verify policies
SELECT * FROM pg_policies WHERE tablename = 'user_credits';
