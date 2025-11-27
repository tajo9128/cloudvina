-- Fix for PGRST205: Could not find the table 'public.user_profiles' in the schema cache

-- 1. Reload PostgREST schema cache
-- This forces Supabase to refresh its knowledge of the database structure
NOTIFY pgrst, 'reload config';

-- 2. Ensure permissions are correct (just in case)
GRANT ALL ON public.user_profiles TO postgres;
GRANT ALL ON public.user_profiles TO service_role;
GRANT ALL ON public.user_profiles TO authenticated;

-- 3. Ensure RLS is enabled but policies exist
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

-- Re-apply policies if they are missing (idempotent-ish)
DROP POLICY IF EXISTS "Users can view own profile" ON public.user_profiles;
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.user_profiles;
CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);
