# quantlib-mm web

Next.js 14 + Tailwind + Recharts showcase frontend for the `quantlib-mm`
Python library.

## Local dev

```bash
cd web
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8765 npm run dev
```

Open <http://localhost:3000>. You'll need the FastAPI backend running on
`8765` for the demo sections to load data (see `api/README.md`).

## Production build (static export)

```bash
cd web
NEXT_PUBLIC_API_URL=https://your-api.example.com npm run build
# Output is in web/out/, suitable for any static host (Render Static Site,
# Vercel static, S3 + CloudFront, etc.).
```

## Stack

- Next.js 14, App Router, `output: "export"` (fully static)
- React 18, TypeScript strict
- Tailwind CSS with custom design tokens (`tailwind.config.ts`)
- Recharts for line/area/scatter/bar charts
- No client-side router data fetching: all demo data comes from the
  FastAPI backend via `lib/api.ts`.
