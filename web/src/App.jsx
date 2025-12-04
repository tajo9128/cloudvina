import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'

// Pages
import TestPage from './pages/TestPage'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import NewJobPage from './pages/NewJobPage'
import JobResultsPage from './pages/JobResultsPage'
// import AdminPage from './pages/AdminPage' // Temporarily disabled
import ConverterPage from './pages/ConverterPage'
import BlogPage from './pages/BlogPage'
import BlogPostPage from './pages/BlogPostPage'
import AboutPage from './pages/AboutPage'
import PrivacyPage from './pages/PrivacyPage'
import TermsPage from './pages/TermsPage'
import ContactPage from './pages/ContactPage'
import AIAnalysisPage from './pages/AIAnalysisPage'
import Layout from './components/Layout'
import AdminLayout from './components/AdminLayout'
import AdminRoute from './components/AdminRoute'
import AdminDashboard from './pages/admin/Dashboard'
import AdminJobs from './pages/admin/Jobs'
import AdminUsers from './pages/admin/Users'

const queryClient = new QueryClient()

function App() {
    const [session, setSession] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        // Get initial session
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session)
            setLoading(false)
        })

        // Listen for auth changes
        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session)
        })

        return () => subscription.unsubscribe()
    }, [])

    useEffect(() => {
        console.log('BioDockify v2.1 Loaded - Build: ' + new Date().toISOString())
    }, [])

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center">
            <div className="text-xl">Loading...</div>
        </div>
    }

    return (
        <QueryClientProvider client={queryClient}>
            <BrowserRouter>
                <Routes>
                    {/* Admin Routes */}
                    <Route path="/admin" element={
                        <AdminRoute>
                            <AdminLayout />
                        </AdminRoute>
                    }>
                        <Route index element={<AdminDashboard />} />
                        <Route path="jobs" element={<AdminJobs />} />
                        <Route path="users" element={<AdminUsers />} />
                    </Route>

                    {/* Public/User Routes with Main Layout */}
                    <Route element={<Layout><Outlet /></Layout>}>
                        <Route path="/test" element={<TestPage />} />
                        <Route path="/" element={<HomePage />} />
                        <Route path="/login" element={<LoginPage />} />
                        <Route
                            path="/dashboard"
                            element={session ? <DashboardPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/dock/new"
                            element={session ? <NewJobPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/dock/:jobId"
                            element={session ? <JobResultsPage /> : <Navigate to="/login" />}
                        />
                        <Route path="/tools/converter" element={<ConverterPage />} />
                        <Route path="/ai-analysis" element={<AIAnalysisPage />} />
                        <Route path="/blog" element={<BlogPage />} />
                        <Route path="/blog/:slug" element={<BlogPostPage />} />
                        <Route path="/about" element={<AboutPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="/contact" element={<ContactPage />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </QueryClientProvider>
    )
}

export default App
