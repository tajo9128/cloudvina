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
import BatchDockingPage from './pages/BatchDockingPage'
import BatchResultsPage from './pages/BatchResultsPage'
import JobResultsPage from './pages/JobResultsPage'
import MolecularDockingPage from './pages/MolecularDockingPage'
import ConverterPage from './pages/ConverterPage'
import TargetPredictionPage from './pages/TargetPredictionPage'
import MDSimulationPage from './pages/MDSimulationPage'
import MDResultsPage from './pages/MDResultsPage'
import LeadOptimizationPage from './pages/LeadOptimizationPage'
import BlogPage from './pages/BlogPage'
import BlogPostPage from './pages/BlogPostPage'
import AboutPage from './pages/AboutPage'
import PrivacyPage from './pages/PrivacyPage'
import TermsPage from './pages/TermsPage'
import ContactPage from './pages/ContactPage'
import PricingPage from './pages/PricingPage'
import RefundsPage from './pages/RefundsPage'
import ProfilePage from './pages/ProfilePage'
import BillingPage from './pages/BillingPage'
import SupportPage from './pages/SupportPage'
import Layout from './components/Layout'
import AdminLayout from './components/AdminLayout'
import AdminRoute from './components/AdminRoute'
import AdminDashboard from './pages/admin/Dashboard'
import AdminJobs from './pages/admin/Jobs'
import AdminUsers from './pages/admin/Users'
import AdminPhases from './pages/admin/PhasesControl'
import AdminPricing from './pages/admin/PricingControl'
import AdminMessages from './pages/admin/Messages'
import AdminCalendar from './pages/admin/Calendar' // Anticipating Calendar next

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
        console.log('BioDockify v3.0 Loaded - Build: ' + new Date().toISOString())
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
                        <Route path="calendar" element={<AdminCalendar />} />
                        <Route path="messages" element={<AdminMessages />} />
                        <Route path="phases" element={<AdminPhases />} />
                        <Route path="pricing" element={<AdminPricing />} />
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
                            element={<Navigate to="/dock/batch" replace />}
                        />
                        <Route
                            path="/dock/batch"
                            element={session ? <BatchDockingPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/dock/batch/:batchId"
                            element={session ? <BatchResultsPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/dock/:jobId"
                            element={session ? <JobResultsPage /> : <Navigate to="/login" />}
                        />
                        <Route path="/tools/converter" element={<ConverterPage />} />
                        <Route
                            path="/tools/prediction"
                            element={session ? <TargetPredictionPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/md-simulation"
                            element={session ? <MDSimulationPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/md-results/:jobId"
                            element={session ? <MDResultsPage /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/leads"
                            element={session ? <LeadOptimizationPage /> : <Navigate to="/login" />}
                        />

                        <Route path="/blog" element={<BlogPage />} />
                        <Route path="/blog/:slug" element={<BlogPostPage />} />
                        <Route path="/about" element={<AboutPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="/pricing" element={<PricingPage />} />
                        <Route path="/refund-policy" element={<RefundsPage />} />
                        <Route path="/refunds" element={<RefundsPage />} />
                        <Route path="/molecular-docking-online" element={<MolecularDockingPage />} />
                        <Route path="/contact" element={<ContactPage />} />

                        {/* User System Routes */}
                        <Route path="/profile" element={session ? <ProfilePage /> : <Navigate to="/login" />} />
                        <Route path="/billing" element={session ? <BillingPage /> : <Navigate to="/login" />} />
                        <Route path="/support" element={session ? <SupportPage /> : <Navigate to="/login" />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </QueryClientProvider>
    )
}

export default App
