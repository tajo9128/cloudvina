import { useState, useEffect } from 'react'
import { API_URL } from '../config'
import { supabase } from '../supabaseClient'

export default function DrugPropertiesPanel({ jobId, smiles }) {
    const [properties, setProperties] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [expanded, setExpanded] = useState(false)

    useEffect(() => {
        if (jobId) {
            fetchJobProperties()
        } else if (smiles) {
            fetchSmilesProperties()
        }
    }, [jobId, smiles])

    const fetchJobProperties = async () => {
        setLoading(true)
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const res = await fetch(`${API_URL}/jobs/${jobId}/drug-properties`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })

            if (res.ok) {
                const data = await res.json()
                if (data.properties) {
                    setProperties(data.properties)
                }
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchSmilesProperties = async () => {
        setLoading(true)
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const res = await fetch(`${API_URL}/molecules/drug-properties`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ smiles })
            })

            if (res.ok) {
                const data = await res.json()
                setProperties(data)
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <div className="flex items-center gap-3">
                    <div className="animate-spin h-5 w-5 border-2 border-primary-500 border-t-transparent rounded-full"></div>
                    <span className="text-slate-500">Analyzing drug-likeness...</span>
                </div>
            </div>
        )
    }

    if (!properties) return null

    const { molecular_properties, lipinski, veber, drug_likeness, pains, admet_links, summary } = properties

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-slate-200">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="font-bold text-slate-900 flex items-center gap-2">
                            💊 Drug-Likeness Analysis
                        </h3>
                        <p className="text-sm text-slate-500 mt-1">
                            ADMET prediction and pharmacokinetic properties
                        </p>
                    </div>
                    <div className={`px-4 py-2 rounded-full font-bold text-sm ${drug_likeness?.category === 'Drug-like' ? 'bg-green-100 text-green-700 border border-green-200' :
                            drug_likeness?.category === 'Moderate' ? 'bg-yellow-100 text-yellow-700 border border-yellow-200' :
                                'bg-red-100 text-red-700 border border-red-200'
                        }`}>
                        {drug_likeness?.score}/100 — {drug_likeness?.category}
                    </div>
                </div>
            </div>

            {/* Quick Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-slate-200 border-b border-slate-200">
                <div className="p-4 text-center">
                    <div className="text-xs text-slate-500 uppercase mb-1">MW</div>
                    <div className={`text-xl font-bold ${molecular_properties?.molecular_weight > 500 ? 'text-red-600' : 'text-slate-900'}`}>
                        {molecular_properties?.molecular_weight}
                    </div>
                </div>
                <div className="p-4 text-center">
                    <div className="text-xs text-slate-500 uppercase mb-1">LogP</div>
                    <div className={`text-xl font-bold ${molecular_properties?.logp > 5 ? 'text-red-600' : 'text-slate-900'}`}>
                        {molecular_properties?.logp}
                    </div>
                </div>
                <div className="p-4 text-center">
                    <div className="text-xs text-slate-500 uppercase mb-1">HBD / HBA</div>
                    <div className="text-xl font-bold text-slate-900">
                        {molecular_properties?.hbd} / {molecular_properties?.hba}
                    </div>
                </div>
                <div className="p-4 text-center">
                    <div className="text-xs text-slate-500 uppercase mb-1">Lipinski</div>
                    <div className={`text-xl font-bold ${lipinski?.passed ? 'text-green-600' : 'text-red-600'}`}>
                        {lipinski?.passed ? '✓ Pass' : `${lipinski?.violations} Violations`}
                    </div>
                </div>
            </div>

            {/* Rule Checks */}
            <div className="p-6 space-y-4">
                {/* Lipinski */}
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-3">
                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${lipinski?.passed ? 'bg-green-500' : 'bg-red-500'
                            }`}>
                            {lipinski?.passed ? '✓' : lipinski?.violations}
                        </span>
                        <div>
                            <div className="font-medium text-slate-900">Lipinski Rule of 5</div>
                            <div className="text-xs text-slate-500">Oral bioavailability predictor</div>
                        </div>
                    </div>
                    {!lipinski?.passed && (
                        <div className="text-xs text-red-600">
                            {lipinski?.violation_details?.join(', ')}
                        </div>
                    )}
                </div>

                {/* Veber */}
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-3">
                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${veber?.passed ? 'bg-green-500' : 'bg-red-500'
                            }`}>
                            {veber?.passed ? '✓' : veber?.violations}
                        </span>
                        <div>
                            <div className="font-medium text-slate-900">Veber Rules</div>
                            <div className="text-xs text-slate-500">Flexibility and polarity</div>
                        </div>
                    </div>
                    {!veber?.passed && (
                        <div className="text-xs text-red-600">
                            {veber?.violation_details?.join(', ')}
                        </div>
                    )}
                </div>

                {/* PAINS */}
                <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-3">
                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${pains?.passed ? 'bg-green-500' : 'bg-orange-500'
                            }`}>
                            {pains?.passed ? '✓' : '!'}
                        </span>
                        <div>
                            <div className="font-medium text-slate-900">PAINS Filter</div>
                            <div className="text-xs text-slate-500">Assay interference compounds</div>
                        </div>
                    </div>
                    {!pains?.passed && (
                        <div className="text-xs text-orange-600">
                            {pains?.num_alerts} alert(s): {pains?.alerts?.slice(0, 2).join(', ')}
                        </div>
                    )}
                </div>
            </div>

            {/* ADMET External Tools */}
            <div className="px-6 pb-6">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                    External ADMET Tools (Free)
                </div>
                <div className="flex flex-wrap gap-2">
                    {admet_links && Object.entries(admet_links).map(([key, tool]) => (
                        <a
                            key={key}
                            href={tool.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-primary-100 text-slate-700 hover:text-primary-700 rounded-full text-sm font-medium transition-colors"
                            title={tool.description}
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                            </svg>
                            {tool.name}
                        </a>
                    ))}
                </div>
            </div>

            {/* Expandable Details */}
            <div className="border-t border-slate-200">
                <button
                    onClick={() => setExpanded(!expanded)}
                    className="w-full px-6 py-3 flex items-center justify-between text-sm font-medium text-slate-600 hover:bg-slate-50"
                >
                    <span>View All Molecular Properties</span>
                    <svg className={`w-5 h-5 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                    </svg>
                </button>

                {expanded && molecular_properties && (
                    <div className="px-6 pb-6 grid grid-cols-3 md:grid-cols-5 gap-4">
                        {Object.entries(molecular_properties).map(([key, value]) => (
                            <div key={key} className="text-center p-2 bg-slate-50 rounded">
                                <div className="text-xs text-slate-500 uppercase">{key.replace(/_/g, ' ')}</div>
                                <div className="font-mono font-bold text-slate-900">{value}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
