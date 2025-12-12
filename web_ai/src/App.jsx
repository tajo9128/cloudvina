import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './Layout';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import QSARPage from './pages/QSARPage';

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/" element={<LandingPage />} />
                    <Route path="/dashboard" element={<DashboardPage />} />
                    <Route path="/project/:projectId" element={<QSARPage />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    );
}

export default App;
