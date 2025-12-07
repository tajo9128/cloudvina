import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { CreditCard, DollarSign, Package } from 'lucide-react';

const PricingSection = ({ title, data, onUpdate }) => {
    const [formData, setFormData] = useState(data);
    const [changed, setChanged] = useState(false);

    useEffect(() => {
        setFormData(data);
    }, [data]);

    const handleChange = (key, value) => {
        setFormData(prev => ({ ...prev, [key]: value }));
        setChanged(true);
    };

    const handleSave = () => {
        onUpdate(formData);
        setChanged(false);
    };

    return (
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-6">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold text-white capitalize">{title.replace('_', ' ')}</h3>
                {changed && (
                    <button
                        onClick={handleSave}
                        className="bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1 rounded text-sm transition-colors"
                    >
                        Save Changes
                    </button>
                )}
            </div>
            <div className="space-y-4">
                {Object.entries(formData).map(([key, value]) => (
                    <div key={key} className="grid grid-cols-2 gap-4 items-center">
                        <label className="text-slate-400 text-sm capitalize">{key.replace(/_/g, ' ')}</label>
                        <input
                            type={typeof value === 'number' ? 'number' : 'text'}
                            step={typeof value === 'number' && key.includes('cost') ? '0.01' : '1'}
                            value={value}
                            onChange={(e) => handleChange(key, typeof value === 'number' ? parseFloat(e.target.value) : e.target.value)}
                            className="bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white text-sm focus:border-indigo-500 outline-none"
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};

const PricingControl = () => {
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const res = await fetch(`${import.meta.env.VITE_API_URL}/admin/settings`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            const data = await res.json();
            setConfig(data.pricing || {});
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const updateSection = async (sectionKey, newData) => {
        const newPricingConfig = { ...config, [sectionKey]: newData };
        setConfig(newPricingConfig); // Optimistic

        try {
            const { data: { session } } = await supabase.auth.getSession();
            await fetch(`${import.meta.env.VITE_API_URL}/admin/settings/pricing`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ [sectionKey]: newData })
            });
        } catch (err) {
            console.error("Failed to update pricing", err);
            fetchSettings();
        }
    };

    if (loading) return <div className="p-8 text-slate-400">Loading pricing configuration...</div>;

    return (
        <div className="p-6 md:p-8 max-w-[1920px] mx-auto space-y-6">
            <header className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                    <DollarSign className="text-green-400" size={32} />
                    <h1 className="text-3xl font-bold text-white">Billing & Pricing</h1>
                </div>
                <p className="text-slate-400">Configure cost models, valid currencies, and quota tiers for the platform.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* General Settings */}
                <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-6 h-fit">
                    <h3 className="text-lg font-bold text-white mb-4">Global Currency</h3>
                    <div className="flex items-center gap-4">
                        <CreditCard className="text-slate-500" />
                        <select
                            value={config.currency || 'USD'}
                            onChange={(e) => updateSection('currency', e.target.value)} // Fix: currency is top-level string potentially, check config structure
                            className="bg-slate-900 border border-slate-700 rounded px-4 py-2 text-white outline-none"
                        >
                            <option value="USD">USD ($)</option>
                            <option value="EUR">EUR (€)</option>
                            <option value="GBP">GBP (£)</option>
                            <option value="INR">INR (₹)</option>
                        </select>
                    </div>
                </div>

                {/* Tiers */}
                {Object.entries(config).map(([key, value]) => {
                    if (typeof value === 'object' && value !== null) {
                        return (
                            <PricingSection
                                key={key}
                                title={key}
                                data={value}
                                onUpdate={(newData) => updateSection(key, newData)}
                            />
                        );
                    }
                    return null;
                })}
            </div>

            <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <p className="text-blue-400 text-sm flex items-center gap-2">
                    <Package size={16} />
                    Note: Adjusting "Free Tier" quotas will immediately affect all non-paying users. Pricing changes only affect display until payment gateway integration.
                </p>
            </div>
        </div>
    );
};

export default PricingControl;
