// Comprehensive mock data for the Community Forum (wpForo style)

export const currentUser = {
    id: 'user_current',
    name: 'Dr. Alex',
    avatar: null,
    role: 'Member',
    reputation: 150
};

export const users = {
    'admin': { id: 'admin', name: 'Admin', role: 'Administrator', avatar: null, reputation: 9999, joinDate: '2023-01-01', posts: 1542 },
    'sarah': { id: 'sarah', name: 'Dr. Sarah', role: 'Moderator', avatar: null, reputation: 2500, joinDate: '2023-03-15', posts: 890 },
    'alex': currentUser,
    'newbie': { id: 'newbie', name: 'BioStudent24', role: 'Member', avatar: null, reputation: 10, joinDate: '2024-11-20', posts: 5 }
};

export const members = Object.values(users);

export const categories = [
    {
        id: 'main',
        title: 'Main Category',
        order: 1,
        forums: [
            {
                id: 1,
                title: 'Announcements & News',
                slug: 'announcements',
                description: 'Official news and updates from the BioDockify team.',
                icon: '📢',
                topicsCount: 15,
                postsCount: 42,
                lastPost: { title: 'v2.0 Release Notes', author: 'Admin', time: '2 hours ago', avatar: null }
            },
            {
                id: 2,
                title: 'General Discussion',
                slug: 'general',
                description: 'Talk about anything related to molecular docking and drug discovery.',
                icon: '💬',
                topicsCount: 128,
                postsCount: 450,
                lastPost: { title: 'Best practices for protein prep', author: 'Dr. Sarah', time: '5 mins ago', avatar: null }
            },
            {
                id: 3,
                title: 'Community Showcase',
                slug: 'showcase',
                description: 'Share your research findings, papers, and success stories.',
                icon: '🏆',
                topicsCount: 45,
                postsCount: 112,
                lastPost: { title: 'My latest publication', author: 'Dr. Alex', time: '1 day ago', avatar: null }
            }
        ]
    },
    {
        id: 'support',
        title: 'Support & Help',
        order: 2,
        forums: [
            {
                id: 4,
                title: 'Q & A',
                slug: 'qa',
                description: 'Ask questions and get answers from the community and experts.',
                icon: '❓',
                topicsCount: 342,
                postsCount: 1205,
                lastPost: { title: 'Error codes in Vina', author: 'BioStudent24', time: '10 mins ago', avatar: null }
            },
            {
                id: 5,
                title: 'Bug Reports',
                slug: 'bugs',
                description: 'Report technical issues and bugs.',
                icon: '🐛',
                topicsCount: 28,
                postsCount: 89,
                lastPost: { title: 'Login issue on mobile', author: 'Admin', time: '3 hours ago', avatar: null }
            }
        ]
    }
];

export const topics = [
    // General Discussion Topics
    {
        id: 101,
        forumId: 2,
        title: 'Best practices for protein preparation before docking',
        slug: 'best-practices-protein-prep',
        author: users['sarah'],
        createdAt: '2024-12-08T10:00:00Z',
        views: 1250,
        replies: 24,
        isSticky: true,
        isLocked: false,
        tags: ['protein-prep', 'tutorial'],
        lastPost: { author: users['newbie'], time: '5 mins ago' }
    },
    {
        id: 102,
        forumId: 2,
        title: 'Comparative analysis of Vina vs. Glide scoring functions',
        slug: 'vina-vs-glide-scoring',
        author: users['admin'],
        createdAt: '2024-12-07T14:30:00Z',
        views: 890,
        replies: 15,
        isSticky: false,
        isLocked: false,
        tags: ['scoring', 'comparison'],
        lastPost: { author: users['sarah'], time: '2 hours ago' }
    },
    {
        id: 103,
        forumId: 2,
        title: 'How to handle flexible residues in the binding pocket?',
        slug: 'flexible-residues-binding-pocket',
        author: users['newbie'],
        createdAt: '2024-12-09T09:15:00Z',
        views: 45,
        replies: 3,
        isSticky: false,
        isLocked: false,
        tags: ['flexible-docking', 'help'],
        lastPost: { author: users['admin'], time: '1 hour ago' }
    },
    // Q&A Topics
    {
        id: 201,
        forumId: 4,
        title: 'Error: "Ligand atom type not found"',
        slug: 'error-ligand-atom-type',
        author: users['newbie'],
        createdAt: '2024-12-09T11:00:00Z',
        views: 12,
        replies: 1,
        isSticky: false,
        isLocked: false,
        isSolved: true,
        tags: ['error', 'troubleshooting'],
        lastPost: { author: users['sarah'], time: '30 mins ago' }
    }
];

export const posts = [
    // Topic 101 Posts
    {
        id: 1001,
        topicId: 101,
        author: users['sarah'],
        content: `
            <p>Protein preparation is the SINGLE most important step in molecular docking. Here is my checklist for a perfect PDB setup:</p>
            <ol>
                <li><strong>Remove Water Molecules:</strong> Unless you know a specific water mediates binding, strip them all.</li>
                <li><strong>Add Hydrogens:</strong> Crystal structures lack H atoms. Add them at pH 7.4.</li>
                <li><strong>Check Protonation States:</strong> His, Asp, Glu states vary by environment. Use PropKa.</li>
                <li><strong>Fix Missing Side Chains:</strong> Use a modeling tool to rebuild missing atoms.</li>
            </ol>
            <p>What tools do you all use? I prefer the automated pipeline in BioDockify Tools.</p>
        `,
        createdAt: '2024-12-08T10:00:00Z',
        likes: 45,
        isAnswer: false
    },
    {
        id: 1002,
        topicId: 101,
        author: users['admin'],
        content: `<p>Great list Sarah! I would also add <strong>charge assignment</strong>. Using Gasteiger charges is standard for Vina.</p>`,
        createdAt: '2024-12-08T10:05:00Z',
        likes: 12,
        replyTo: 1001
    },
    {
        id: 1003,
        topicId: 101,
        author: users['newbie'],
        content: `<p>Thanks for this! I was forgetting to check protonation states. That explains my weird results.</p>`,
        createdAt: '2024-12-09T12:00:00Z',
        likes: 3,
        replyTo: 1001
    }
];

export const recentTopics = topics.slice(0, 5);

export const stats = {
    members: 1250,
    online: 45,
    totalPosts: 5432,
    totalTopics: 1200
};
