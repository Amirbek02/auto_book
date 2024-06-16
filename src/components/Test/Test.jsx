import React from 'react';
import { Link } from 'react-router-dom';

import './tests.scss';

function Test() {
  return (
    <div className="test">
      <Link className="test__item" to="/test1">
        <img className="test__img" src="test.svg" alt="" />
        <p className="test__title">Тест</p>
      </Link>
    </div>
  );
}

export default Test;
