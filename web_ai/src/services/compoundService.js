import axios from 'axios';
import { API_BASE_URL } from '../config';
import { supabase } from '../supabaseClient';
import Papa from 'papaparse';

const getHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
    };
};

export const compoundService = {
    // Upload compounds (CSV Parsing + API Call)
    async uploadCompounds(projectId, csvText) {
        return new Promise((resolve, reject) => {
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: async (results) => {
                    const rows = results.data;
                    if (!rows || rows.length === 0) {
                        reject(new Error("CSV is empty"));
                        return;
                    }

                    // Simple Validation
                    const hasSmiles = Object.keys(rows[0]).some(k => k.toLowerCase() === 'smiles');
                    if (!hasSmiles) {
                        reject(new Error("CSV must contain a 'smiles' column"));
                        return;
                    }

                    // Format for API
                    const compounds = rows.map(row => {
                        // Find key that matches 'smiles' case-insensitively
                        const smilesKey = Object.keys(row).find(k => k.toLowerCase() === 'smiles');
                        return {
                            project_id: projectId,
                            smiles: row[smilesKey],
                            chem_name: row['name'] || row['Name'] || null,
                            properties: row
                        };
                    });

                    try {
                        const headers = await getHeaders();
                        const response = await axios.post(`${API_BASE_URL}/compounds/`, {
                            project_id: projectId,
                            compounds
                        }, { headers });
                        resolve(response.data);
                    } catch (e) {
                        reject(e);
                    }
                },
                error: (err) => reject(err)
            });
        });
    },

    // Get compounds for a project
    async getCompounds(projectId) {
        const headers = await getHeaders();
        const response = await axios.get(`${API_BASE_URL}/compounds/${projectId}`, { headers });
        return response.data;
    }
};
