const action = require('./action');
const fs = require('fs');
const core = require('@actions/core');

const version = fs.readFileSync(core.getInput('versionFile'), 'utf8');

const newVersion = action.bumpVersion(
  version,
  core.getInput('bumpType'),
  core.getInput('prereleaseSuffix')
);

fs.writeFileSync(core.getInput('versionFile'), newVersion);