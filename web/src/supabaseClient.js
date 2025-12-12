import { createClient } from '@supabase/supabase-js'

// TEMPORARY FIX: Hardcoded values because Vercel env vars not working
// TODO: Revert this once Vercel environment variables are properly configured
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://ohzfktmtwmubyhvspexv.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oemZrdG10d211YnlodnNwZXh2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM4MTk2MjgsImV4cCI6MjA3OTM5NTYyOH0.v8qHeRx5jkL8iaNEbEP_NMIvvUk4oidwwW6PkXo_DVY'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
