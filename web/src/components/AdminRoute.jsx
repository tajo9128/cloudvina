import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { supabase } from '../supabaseClient';

const AdminRoute = ({ children }) => {
    const [isAdmin, setIsAdmin] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        checkAdminStatus();
    }, []);

    const checkAdminStatus = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();

            if (!session) {
                setIsAdmin(false);
                return;
            }

            // Check if user is admin
            const { data: profile } = await supabase
                .from('profiles')
                .select('is_admin')
                .eq('id', session.user.id)
                .single();

            setIsAdmin(profile?.is_admin || false);
        } catch (error) {
            console.error('Error checking admin status:', error);
            setIsAdmin(false);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-[#0B1121]">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
            </div>
        );
    }

    if (!isAdmin) {
        return <Navigate to="/" replace />;
    }

    return <>{children}</>;
};

export default AdminRoute;
