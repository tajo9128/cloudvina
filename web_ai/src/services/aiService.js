import axios from 'axios';
import { API_BASE_URL, SUPABASE_URL, SUPABASE_ANON_KEY } from '../config';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const getHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
    };
};

export const trainModel = async (projectId, modelName, smiles, targets) => {
    const headers = await getHeaders();
    // Updated path: /qsar/train
    return axios.post(`${API_BASE_URL}/qsar/train`, {
        project_id: projectId,
        model_name: modelName,
        target_column: "activity", // hardcoded for demo
        smiles,
        targets
    }, { headers });
};

export const predictModel = async (modelPath, smiles) => {
    const headers = await getHeaders();
    // Updated path: /qsar/predict
    return axios.post(`${API_BASE_URL}/qsar/predict`, {
        model_path: modelPath,
        smiles
    }, { headers });
};

/**
 * Predict bioactivity using pre-trained ChemBERTa models.
 * @param {string[]} smilesList - Array of SMILES strings
 * @param {string} diseaseTarget - One of: alzheimers, cancer, diabetes, parkinson, cardiovascular
 * @returns {Promise} - Predictions with scores and interpretations
 */
export const predictByDisease = async (smilesList, diseaseTarget = 'alzheimers') => {
    const headers = await getHeaders();
    export const aiService = {
        trainModel: async (projectId, modelName, smiles, targets) => {
            const headers = await getHeaders();
            // Updated path: /qsar/train
            return axios.post(`${API_BASE_URL}/qsar/train`, {
                project_id: projectId,
                model_name: modelName,
                target_column: "activity", // hardcoded for demo
                smiles,
                targets
            }, { headers });
        },

        predictModel: async (modelPath, smiles) => {
            const headers = await getHeaders();
            // Updated path: /qsar/predict
            return axios.post(`${API_BASE_URL}/qsar/predict`, {
                model_path: modelPath,
                smiles
            }, { headers });
        },

        /**
         * Predict bioactivity using pre-trained ChemBERTa models.
         * @param {string[]} smilesList - Array of SMILES strings
         * @param {string} diseaseTarget - One of: alzheimers, cancer, diabetes, parkinson, cardiovascular
         * @returns {Promise} - Predictions with scores and interpretations
         */
        predictByDisease: async (smilesList, diseaseTarget = 'alzheimers') => {
            const headers = await getHeaders();
            return axios.post(`${API_BASE_URL}/qsar/predict/disease`, {
                smiles: smilesList,
                disease_target: diseaseTarget
            }, { headers });
        },

        /**
         * Train Auto-QSAR Model
         * @param {FormData} formData - Contains 'file'
         * @param {string} targetColumn 
         * @param {string} modelName 
         */
        trainAutoQSAR: async (formData, targetColumn, modelName) => {
            // Backend endpoint expects params for column/name, file in body
            const url = `${API_BASE_URL}/qsar/train/auto?target_column=${targetColumn}&model_name=${modelName}`;
            const response = await axios.post(url, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;
        },

        /**
         * Predict Toxicity
         * @param {string[]} smilesList 
         */
        predictToxicity: async (smilesList) => {
            const response = await axios.post(`${API_BASE_URL}/toxicity/predict`, {
                smiles: smilesList
            });
            return response.data;
        },

        /**
         * Get available disease targets
         */
        getDiseases: () => {
            return [
                { id: "alzheimers", name: "Alzheimer's (BACE1)" },
                { id: "cancer", name: "Cancer (TP53)" },
                { id: "diabetes", name: "Diabetes (SGLT2)" },
                { id: "parkinson", name: "Parkinson's (LRRK2)" },
                { id: "cardiovascular", name: "Cardiovascular (HMGCR)" }
            ];
        },
    };
