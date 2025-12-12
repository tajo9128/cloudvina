import React from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import { Outlet } from 'react-router-dom';

export default function Layout({ children }) {
    return (
        <div className="min-h-screen flex flex-col bg-slate-50">
            <Header />
            <main className="flex-grow">
                {/* Outlet for nested routes, or children for direct wrapping */}
                {children ? children : <Outlet />}
            </main>
            <Footer />
        </div>
    );
}
