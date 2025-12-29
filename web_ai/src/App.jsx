import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './Layout';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import QSARPage from './pages/QSARPage';
import FormulationPage from './pages/FormulationPage';
import PlanAPage from './pages/PlanAPage';
import PlanBPage from './pages/PlanBPage';

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/" element={<LandingPage />} />
                    <Route path="/dashboard" element={<DashboardPage />} />
                    <Route path="/project/cns" element={<PlanAPage />} />
                    <Route path="/project/phytochemicals" element={<PlanBPage />} />
                    <Route path="/project/:projectId" element={<QSARPage />} />
                    <Route path="/formulation" element={<FormulationPage />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    );
}

export default App;
