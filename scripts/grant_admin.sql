-- Grant Admin Access to specific users
-- Run this in the Supabase SQL Editor

UPDATE profiles
SET 
    is_admin = true, 
    role = 'admin',
    updated_at = NOW()
WHERE email IN (
    'biodockify@hotmail.com', 
    'cloudvina2025@gmail.com'
);

-- Verify the changes
SELECT id, email, role, is_admin 
FROM profiles 
WHERE email IN (
    'biodockify@hotmail.com', 
    'cloudvina2025@gmail.com'
);
