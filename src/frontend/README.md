## Frontend (React) – Setup and Usage

### Prerequisites
- Node.js 18 LTS or 20+ (recommended)
- npm 9+ (bundled with Node)

Check your versions:
```
node -v
npm -v
```

### Install dependencies
```
cd src\frontend
npm install
```


### Run the dev server
```
cd src\frontend
npm start
```

Then open http://localhost:3000.

### Build for production
```
npm run build
```

Outputs an optimized `build/` directory.

### Available scripts
```
npm start      # start dev server (http://localhost:3000)
npm test       # run tests in watch mode
npm run build  # production build to ./build
npm run eject  # CRA eject (irreversible)
```