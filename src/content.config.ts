import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
    loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
    schema: z.object({
        title: z.string(),
        description: z.string(),
        date: z.coerce.date(),
        draft: z.boolean().default(false),
        tags: z.array(z.string()).optional(),
        author: z.union([
            z.string(),
            z.object({
                name: z.string(),
                url: z.string().url().optional(),
            }),
        ]).optional(),
    }),
});

export const collections = { blog };

