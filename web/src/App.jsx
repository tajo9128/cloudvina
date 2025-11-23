import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'

// Pages (to be created)
import HomePage from './pages/HomePage'

export default App
