import { createClient } from '@supabase/supabase-js'

// TEMPORARY FIX: Hardcoded values because Vercel env vars not working
// TODO: Revert this once Vercel environment variables are properly configured
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || ''
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || ''

if (!supabaseUrl || !supabaseAnonKey) {
    console.warn('Supabase credentials missing! Check your .env file or deployment settings.')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
