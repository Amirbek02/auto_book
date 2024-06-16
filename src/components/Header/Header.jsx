import React from 'react';

import './header.scss';
import { Link } from 'react-router-dom';
function Header() {
  return (
    <header className="header">
      <div className="container header__container">
        <Link to="/">
          <div className="headerRight">
            <img src="auto_icon.png" alt="" className="header__logo" />
            <div className="header__logo-texts">
              <h2 className="header__title">Машиналық</h2>
              <p className="header__descr">ОҚЫТУДЫҢ НЕГІЗДЕРІ</p>
            </div>
          </div>
        </Link>
        <div className="headerLeft">
          <ul className="menu">
            <li className="menu__item">
              <a href="/" className="menu__link">
                Негізгі бет
              </a>
            </li>
            <li className="menu__item">
              <Link to="/read" className="menu__link">
                Электронды оқулық
              </Link>
            </li>
            <li className="menu__item">
              <Link to="/work" className="menu__link">
                Зертханалық жұмыс
              </Link>
            </li>
            <li className="menu__item">
              <Link to="/test" className="menu__link">
                Тест
              </Link>
            </li>
            <li className="menu__item">
              <Link to="/about" className="menu__link">
                Автор туралы
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </header>
  );
}

export default Header;
