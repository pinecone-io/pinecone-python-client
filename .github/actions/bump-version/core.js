// Copied these commands out of the github actions toolkit
// because actually depending on @actions/core requires me to check
// in node_modules and 34MB of dependencies, which I don't want to do.

const fs = require('fs');
const os = require('os');

function getInput(name, options) {
    const val =
      process.env[`INPUT_${name.replace(/ /g, '_').toUpperCase()}`] || ''
    if (options && options.required && !val) {
      throw new Error(`Input required and not supplied: ${name}`)
    }
  
    if (options && options.trimWhitespace === false) {
      return val
    }
  
    return val.trim()
}

function toCommandValue(input) {
    if (input === null || input === undefined) {
      return ''
    } else if (typeof input === 'string' || input instanceof String) {
      return input
    }
    return JSON.stringify(input)
}

function prepareKeyValueMessage(key, value) {
    const delimiter = `delimiter_${Math.floor(Math.random()*100000)}`
    const convertedValue = toCommandValue(value)
  
    // These should realistically never happen, but just in case someone finds a
    // way to exploit uuid generation let's not allow keys or values that contain
    // the delimiter.
    if (key.includes(delimiter)) {
      throw new Error(
        `Unexpected input: name should not contain the delimiter "${delimiter}"`
      )
    }
  
    if (convertedValue.includes(delimiter)) {
      throw new Error(
        `Unexpected input: value should not contain the delimiter "${delimiter}"`
      )
    }
  
    return `${key}<<${delimiter}${os.EOL}${convertedValue}${os.EOL}${delimiter}`
}

function setOutput(name, value) {
    const filePath = process.env['GITHUB_OUTPUT'] || ''
    if (filePath) {
      return issueFileCommand('OUTPUT', prepareKeyValueMessage(name, value))
    }
  
    process.stdout.write(os.EOL)
    issueCommand('set-output', {name}, toCommandValue(value))
}

function issueFileCommand(command, message) {
    const filePath = process.env[`GITHUB_${command}`]
    if (!filePath) {
        throw new Error(
        `Unable to find environment variable for file command ${command}`
        )
    }
    if (!fs.existsSync(filePath)) {
        throw new Error(`Missing file at path: ${filePath}`)
    }

    fs.appendFileSync(filePath, `${toCommandValue(message)}${os.EOL}`, {
        encoding: 'utf8'
    })
}

module.exports = { getInput, setOutput }