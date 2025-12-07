import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './admin/Sidebar';
import Header from './admin/Header';

const AdminLayout = () => {
    const [sidebarOpen, setSidebarOpen] = useState(false);

    return (
        <div className="bg-slate-950 text-white h-screen overflow-hidden flex">
            {/* Sidebar */}
            <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

            {/* Content Area */}
            <div className="relative flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
                {/* Header */}
                <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

                {/* Main Content */}
                <main>
                    <div className="mx-auto max-w-screen-2xl p-4 md:p-6 2xl:p-10">
                        <Outlet />
                    </div>
                </main>
            </div>
        </div>
    );
};

export default AdminLayout;
