const action = require('./action')
const core = require('./core');

jest.mock('./core');

describe('bump-version', () => {
    test('bump major', () => {
      action.bumpVersion('1.2.3', 'major', '')

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('version', '2.0.0');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v2.0.0')
    })

    test('bump minor: existing minor and patch', () => {
      action.bumpVersion('1.2.3', 'minor', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.3.0');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.3.0')
    })

    test('bump minor: with no patch', () => {
      action.bumpVersion('1.2.0', 'minor', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.0')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.0')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.3.0');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.3.0')
    })

    test('bump minor: from existing patch', () => {
      action.bumpVersion('2.2.3', 'minor', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '2.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v2.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('version', '2.3.0');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v2.3.0')
    })

    test('bump patch: existing patch', () => {
      action.bumpVersion('1.2.3', 'patch', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.3')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.2.4');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.2.4')
    })

    test('bump patch: minor with no patch', () => {
      action.bumpVersion('1.2.0', 'patch', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.0')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.0')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.2.1');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.2.1')
    })

    test('bump patch: major with no minor or patch', () => {
      action.bumpVersion('1.0.0', 'patch', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.0.0')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.0.0')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.0.1');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.0.1')
    })

    test('bump patch: major with minor', () => {
      action.bumpVersion('1.1.0', 'patch', '');

      expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.1.0')
      expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.1.0')
      expect(core.setOutput).toHaveBeenCalledWith('version', '1.1.1');
      expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.1.1')
    })

   test('prerelease suffix provided', () => {
    action.bumpVersion('1.2.3', 'patch', 'rc1');

    expect(core.setOutput).toHaveBeenCalledWith('previous_version', '1.2.3')
    expect(core.setOutput).toHaveBeenCalledWith('previous_version_tag', 'v1.2.3')
    expect(core.setOutput).toHaveBeenCalledWith('version', '1.2.4.rc1');
    expect(core.setOutput).toHaveBeenCalledWith('version_tag', 'v1.2.4.rc1')
   })
})