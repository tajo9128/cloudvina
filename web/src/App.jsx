import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'

// Pages (to be created)
import HomePage from './pages/HomePage'
import TermsPage from './pages/TermsPage'
import ContactPage from './pages/ContactPage'

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

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center">
            <div className="text-xl">Loading...</div>
        </div>
    }

    return (
        <QueryClientProvider client={queryClient}>
            <BrowserRouter>
                <Layout>
                    <Routes>
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
                        <Route path="/admin" element={session ? <AdminPage /> : <Navigate to="/login" />} />
                        <Route path="/tools/converter" element={<ConverterPage />} />
                        <Route path="/blog" element={<BlogPage />} />
                        <Route path="/about" element={<AboutPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="/contact" element={<ContactPage />} />
                    </Routes>
                </Layout>
            </BrowserRouter>
        </QueryClientProvider>
    )
}

export default App
