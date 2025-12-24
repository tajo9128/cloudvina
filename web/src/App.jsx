import { BrowserRouter, Routes, Route, Navigate, Outlet, useLocation, useParams } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'
import { initAnalytics, identifyUser, trackEvent } from './services/analytics';

// Pages
import TestPage from './pages/TestPage'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import NewJobPage from './pages/NewJobPage'
import BatchDockingPage from './pages/BatchDockingPage'
import BatchResultsPage from './pages/BatchResultsPage'
import JobAnalysisPage from './pages/JobAnalysisPage'
import JobResultsPage from './pages/JobResultsPage'
import MolecularDockingPage from './pages/MolecularDockingPage'
import ConverterPage from './pages/ConverterPage'
import TargetPredictionPage from './pages/TargetPredictionPage'
import MDSimulationPage from './pages/MDSimulationPage'
import MDResultsPage from './pages/MDResultsPage'
import MDStabilityPage from './pages/MDStabilityPage' // NEW: Isolated MD Page
import OnboardingPage from './pages/OnboardingPage' // [NEW]
import BenchmarkingPage from './pages/tools/BenchmarkingPage' // [NEW] Sprint 3.1
import ThreeDViewerPage from './pages/tools/ThreeDViewerPage'
import AdmetToolPage from './pages/tools/AdmetToolPage'
import LeadOptimizationPage from './pages/LeadOptimizationPage'
import BlogPage from './pages/BlogPage'
import BlogPostPage from './pages/BlogPostPage'
import AboutPage from './pages/AboutPage'
import PrivacyPage from './pages/PrivacyPage'
import TermsPage from './pages/TermsPage'
import ContactPage from './pages/ContactPage'
import PricingPage from './pages/PricingPage'
import DeveloperPage from './pages/DeveloperPage' // [NEW] Sprint 4
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
import FDACompliancePage from './pages/admin/FDACompliancePage' // [NEW] Sprint 2.1
import RBACManagerPage from './pages/admin/RBACManagerPage' // [NEW] Sprint 2.2

const queryClient = new QueryClient()

// 50: const queryClient = new QueryClient()

function AppRoutes() {
    const [session, setSession] = useState(null)
    const [isAdmin, setIsAdmin] = useState(false);
    const [loading, setLoading] = useState(true)
    const location = useLocation(); // Now safe to use here

    useEffect(() => {
        // Initialize Analytics Safe Mode
        initAnalytics();

        // Get initial session
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session)
            if (session) {
                checkAdmin(session.user.id);
                identifyUser(session.user.id, { email: session.user.email }); // Track User
            }
            setLoading(false)
        })

        // Listen for auth changes
        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session)
            if (session) {
                checkAdmin(session.user.id);
                identifyUser(session.user.id, { email: session.user.email });
            } else {
                setIsAdmin(false);
            }
            setLoading(false);
        })

        return () => subscription.unsubscribe()
    }, [])

    // Track Page Views
    useEffect(() => {
        trackEvent('page:viewed', { path: location.pathname });
    }, [location]);

    useEffect(() => {
        console.log('BioDockify v3.2 (Router in Main) Loaded - Build: ' + new Date().toISOString())
    }, [])

    const checkAdmin = async (userId) => {
        // ... existing admin check logic placeholder
    };

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center">
            <div className="text-xl">Loading...</div>
        </div>
    }

    return (
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
                <Route path="fda" element={<FDACompliancePage />} />
                <Route path="roles" element={<RBACManagerPage />} />
            </Route>

            {/* Public/User Routes with Main Layout */}
            <Route element={<Layout><Outlet /></Layout>}>
                <Route path="/test" element={<TestPage />} />
                <Route path="/" element={<HomePage />} />
                <Route path="/login" element={<LoginPage />} />
                {/* [NEW] Onboarding */}
                <Route path="/onboarding" element={<OnboardingPage />} />
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
                {/* Fix: Redirect direct /batch/ links to /dock/batch/ properly using inline component */}
                <Route
                    path="/batch/:batchId"
                    element={<RedirectToNewBatchUrl />}
                />

                <Route
                    path="/dock/batch/:batchId"
                    element={session ? <BatchResultsPage /> : <Navigate to="/login" />}
                />
                {/* Note: Navigate's :batchId param substitution relies on matching param name in path? 
                    Actually, react-router Navigate with params needs a relative path or smart component.
                    Better to just use an inline component that reads params and redirects. 
                    Or simply map /batch/:batchId to the component if we want to support it directly.
                    Let's just map it to BatchResultsPage directly to be safe and fast.
                */}
                <Route
                    path="/batch/:batchId"
                    element={session ? <BatchResultsPage /> : <Navigate to="/login" />}
                />
                <Route
                    path="/jobs/:jobId/analysis"
                    element={session ? <JobAnalysisPage /> : <Navigate to="/login" />}
                />
                <Route
                    path="/dock/:jobId"
                    element={session ? <JobResultsPage /> : <Navigate to="/login" />}
                />
                <Route path="/tools/converter" element={<ConverterPage />} />
                <Route path="/3d-viewer" element={<ThreeDViewerPage />} />
                <Route path="/tools/admet" element={<AdmetToolPage />} />
                <Route
                    path="/tools/prediction"
                    element={session ? <TargetPredictionPage /> : <Navigate to="/login" />}
                />
                <Route
                    path="/tools/benchmark"
                    element={session ? <BenchmarkingPage /> : <Navigate to="/login" />}
                />
                <Route
                    path="/md-simulation"
                    element={session ? <MDSimulationPage /> : <Navigate to="/login" />}
                />
                <Route
                    path="/md-results/:jobId"
                    element={session ? <MDResultsPage /> : <Navigate to="/login" />}
                />
                {/* ISOLATED MD STABILITY ROUTE */}
                <Route
                    path="/md-analysis"
                    element={session ? <MDStabilityPage /> : <Navigate to="/login" />}
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
                <Route path="/developer" element={<DeveloperPage />} />
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
    )
}

// Helper for dynamic redirects
function RedirectToNewBatchUrl() {
    const { batchId } = useParams()
    return <Navigate to={`/dock/batch/${batchId}`} replace />
}

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <AppRoutes />
        </QueryClientProvider>
    )
}

export default App
