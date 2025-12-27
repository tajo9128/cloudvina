-- 1. Fix Missing Schema Columns
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS credits INTEGER DEFAULT 10;

ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS plan TEXT DEFAULT 'free';

ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT false;

ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user';

-- 2. Force Insert/Update Admin User
-- This assumes the user exists in auth.users. 
-- If 'biodockify@hotmail.com' hasn't signed up, this part will just do nothing (which is fine).
INSERT INTO public.profiles (id, email, role, is_admin, credits, plan)
SELECT 
    id, 
    email, 
    'admin', 
    true, 
    1000, 
    'premium'
FROM auth.users
WHERE email = 'biodockify@hotmail.com'
ON CONFLICT (id) DO UPDATE
SET 
    is_admin = true, 
    role = 'admin',
    credits = 1000,
    plan = 'premium';

-- 3. Verify
SELECT * FROM public.profiles WHERE email = 'biodockify@hotmail.com';
