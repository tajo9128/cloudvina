import React from 'react';

export default function ModelStats({ models }) {
    return (
        <div className="space-y-4">
            <h3 className="font-semibold text-slate-600">Available Models</h3>
            {models.length === 0 ? (
                <p className="text-slate-400 text-sm italic">No models trained yet.</p>
            ) : (
                models.map(m => (
                    <div key={m.id} className="bg-white p-4 rounded-lg border border-slate-200 hover:border-indigo-400 cursor-pointer transition-colors shadow-sm">
                        <div className="flex justify-between items-start mb-2">
                            <h4 className="font-bold text-slate-900">{m.name}</h4>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${m.status === 'ready' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                                {m.status}
                            </span>
                        </div>
                        <div className="text-xs text-slate-500">
                            R²: {m.metrics?.r2?.toFixed(3) || 'N/A'} • RMSE: {m.metrics?.rmse?.toFixed(3) || 'N/A'}
                        </div>
                    </div>
                ))
            )}
        </div>
    );
}
