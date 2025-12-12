import React from 'react';
import { BrowserRouter as Router, Routes, Route, Outlet } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import BlogPage from './pages/BlogPage';
import BlogPostPage from './pages/BlogPostPage';

import CoursesPage from './pages/CoursesPage';
import CourseDetailPage from './pages/CourseDetailPage';
import LessonPage from './pages/LessonPage';
import CertificatePage from './pages/CertificatePage';
import CommunityPage from './pages/CommunityPage';
import MembersPage from './pages/MembersPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import LoginPage2 from './pages/LoginPage2';
import SignupPage2 from './pages/SignupPage2';
import ForumPage from './pages/ForumPage';
import TopicPage from './pages/TopicPage';
import ProfilePage from './pages/ProfilePage';
import AdminLayout from './layouts/AdminLayout';
import AdminDashboardPage from './pages/admin/AdminDashboardPage';
import AdminCoursesPage from './pages/admin/AdminCoursesPage';
import AdminCourseEditorPage from './pages/admin/AdminCourseEditorPage';
import AdminUsersPage from './pages/admin/AdminUsersPage';

const ProfilePage = () => <div className="min-h-screen p-8"><h1 className="text-3xl font-bold">Profile - Coming Soon</h1></div>;
const SettingsPage = () => <div className="min-h-screen p-8"><h1 className="text-3xl font-bold">Settings - Coming Soon</h1></div>;
const SupportPage = () => <div className="min-h-screen p-8"><h1 className="text-3xl font-bold">Support - Coming Soon</h1></div>;

const MainLayout = () => {
    return (
        <div className="min-h-screen flex flex-col bg-white">
            <Header />
            <main className="flex-1">
                <Outlet />
            </main>
            <Footer />
        </div>
    );
};

export default function App() {
    return (
        <AuthProvider>
            <Router>
                <Routes>
                    {/* Auth Routes (No Header/Footer) */}
                    <Route path="/login" element={<LoginPage />} />
                    <Route path="/signup" element={<SignupPage />} />
                    <Route path="/login2" element={<LoginPage2 />} />
                    <Route path="/signup2" element={<SignupPage2 />} />

                    {/* Main Application Routes (With Header/Footer) */}
                    <Route element={<MainLayout />}>
                        <Route path="/" element={<HomePage />} />
                        <Route path="/dashboard" element={<DashboardPage />} />
                        <Route path="/blog" element={<BlogPage />} />
                        <Route path="/blog/:slug" element={<BlogPostPage />} />

                        <Route path="/courses" element={<CoursesPage />} />
                        <Route path="/courses/:slug" element={<CourseDetailPage />} />
                        <Route path="/courses/:slug/lessons/:lessonId" element={<LessonPage />} />
                        <Route path="/courses/:slug/certificate" element={<CertificatePage />} />

                        <Route path="/community" element={<CommunityPage />} />
                        <Route path="/community/members" element={<MembersPage />} />
                        <Route path="/community/forum/:forumId" element={<ForumPage />} />
                        <Route path="/community/topic/:topicId" element={<TopicPage />} />
                        <Route path="/community/profile/:username" element={<ProfilePage />} />

                        <Route path="/profile" element={<ProfilePage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                        <Route path="/support" element={<SupportPage />} />
                    </Route>

                    {/* Admin Routes (Dedicated Layout) */}
                    <Route path="/admin" element={<AdminLayout />}>
                        <Route index element={<AdminDashboardPage />} />
                        <Route path="users" element={<AdminUsersPage />} />
                        <Route path="courses" element={<AdminCoursesPage />} />
                        <Route path="courses/new" element={<AdminCourseEditorPage />} />
                        <Route path="courses/:courseId/edit" element={<AdminCourseEditorPage />} />
                    </Route>
                </Routes>
            </Router>
        </AuthProvider>
    );
}
