import React from 'react';
import { Upload } from 'lucide-react';

export default function CSVUploader({ onUpload }) {
    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            const text = event.target.result;
            onUpload(text);
        };
        reader.readAsText(file);
    };

    return (
        <div className="bg-white border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:border-indigo-500 transition-colors mb-8">
            <Upload className="mx-auto h-10 w-10 text-slate-400 mb-3" />
            <h3 className="text-sm font-medium text-slate-900">Upload CSV Dataset</h3>
            <p className="text-xs text-slate-500 mb-4">Must contain "smiles" column and target values.</p>
            <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            />
        </div>
    );
}
