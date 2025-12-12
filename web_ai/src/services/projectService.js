import axios from 'axios';
import { API_BASE_URL } from '../config';
import { supabase } from '../supabaseClient';

const getHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session?.access_token}`
    };
};

export const projectService = {
    // List all projects for current user
    async getProjects() {
        const headers = await getHeaders();
        const response = await axios.get(`${API_BASE_URL}/projects/`, { headers });
        return response.data;
    },

    // Create a new project
    async createProject(name, description) {
        const headers = await getHeaders();
        const response = await axios.post(`${API_BASE_URL}/projects/`, {
            name,
            description
        }, { headers });
        return response.data;
    },

    // Get single project details
    async getProjectDetails(projectId) {
        const headers = await getHeaders();
        const response = await axios.get(`${API_BASE_URL}/projects/${projectId}`, { headers });
        return response.data;
    },

    // Delete project
    async deleteProject(projectId) {
        const headers = await getHeaders();
        await axios.delete(`${API_BASE_URL}/projects/${projectId}`, { headers });
        return true;
    }
};
