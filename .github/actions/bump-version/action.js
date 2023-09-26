const core = require('./core');

function bumpVersion(currentVersion, bumpType, prerelease) {
  let newVersion = calculateNewVersion(currentVersion, bumpType);

  if (prerelease) {
    newVersion = `${newVersion}.${prerelease}`;
  }
  core.setOutput('previous_version', currentVersion)
  core.setOutput('previous_version_tag', `v${currentVersion}`)
  core.setOutput('version', newVersion);
  core.setOutput('version_tag', `v${newVersion}`)

  return newVersion;
}

function calculateNewVersion(currentVersion, bumpType) {
  const [major, minor, patch] = currentVersion.split('.');
  let newVersion;

  switch (bumpType) {
    case 'major':
      newVersion = `${parseInt(major) + 1}.0.0`;
      break;
    case 'minor':
      newVersion = `${major}.${parseInt(minor) + 1}.0`;
      break;
    case 'patch':
      newVersion = `${major}.${minor}.${parseInt(patch) + 1}`;
      break;
    default:
      throw new Error(`Invalid bumpType: ${bumpType}`);
  }

  return newVersion;
}

module.exports = { bumpVersion }