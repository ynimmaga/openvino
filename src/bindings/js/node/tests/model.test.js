// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('..');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');
const testXml = getModelPath().xml;
const core = new ov.Core();
const model = core.readModelSync(testXml);

describe('Node.js Model.isDynamic()', () => {
  it('should return a boolean value indicating if the model is dynamic', () => {
    const result = model.isDynamic();
    assert.strictEqual(
      typeof result,
      'boolean',
      'isDynamic() should return a boolean value'
    );
  });

  it('should not accept any arguments', () => {
    assert.throws(
      () => {
        model.isDynamic('unexpected argument');
      },
      /^Error: isDynamic\(\) does not accept any arguments\.$/,
      'Expected isDynamic to throw an error when called with arguments'
    );
  });

  it('returns false for a static model', () => {
    const expectedStatus = false;
    assert.strictEqual(
      model.isDynamic(),
      expectedStatus,
      'Expected isDynamic to return false for a static model'
    );
  });
});

describe('Node.js getFriendlyName() / setFriendlyName()', () => {
  describe('getFriendlyName()', () => {
    it('returns the unique name of the model if no friendly name is set', () => {
      const expectedName = 'test_model';
      assert.strictEqual(model.getFriendlyName(), expectedName);
    });
    it('throws an error when called with arguments', () => {
      assert.throws(
        () => model.getFriendlyName('unexpected argument'),
        /getFriendlyName\(\) does not take any arguments/
      );
    });
  });
  describe('setFriendlyName()', () => {
    it('sets a friendly name for the model', () => {
      assert.doesNotThrow(() => model.setFriendlyName('MyFriendlyName'));
    });

    it('throws an error when called without a string argument', () => {
      assert.throws(
        () => model.setFriendlyName(),
        /Expected a single string argument for the friendly name/
      );
      assert.throws(
        () => model.setFriendlyName(123),
        /Expected a single string argument for the friendly name/
      );
    });

    it('throws an error when called with multiple arguments', () => {
      assert.throws(
        () => model.setFriendlyName('Name1', 'Name2'),
        /Expected a single string argument for the friendly name/
      );
    });

    it('returns the set friendly name of the model', () => {
      const friendlyName = 'MyFriendlyModel';
      model.setFriendlyName(friendlyName);
      assert.strictEqual(model.getFriendlyName(), friendlyName);
    });

    it('retains the last set friendly name when set multiple times', () => {
      model.setFriendlyName('InitialName');
      model.setFriendlyName('FinalName');
      assert.strictEqual(model.getFriendlyName(), 'FinalName');
    });

    it('handles setting an empty string as a friendly name', () => {
      assert.doesNotThrow(() => model.setFriendlyName(''));
      assert.strictEqual(model.getFriendlyName(), 'Model1');
    });
  });
});
