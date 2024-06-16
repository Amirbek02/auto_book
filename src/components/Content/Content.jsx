import React from 'react';

import './content.scss';
import { Link } from 'react-router-dom';

function Content() {
  return (
    <div className="content">
      <div className="container">
        <div className="content__container">
          <div className="content__texts">
            <h1 className="content__title">МАШИНАЛЫҚ ОҚЫТУДЫҢ НЕГІЗДЕРІ</h1>
            <p className="content__descr">
              Машиналық оқытудың негзігі ұғымдарымен, машиналық оқытудың көптеген соңғы танымал
              алгоритмдерін теориялық және практикалық Python қолданбаларымен мысалдар көрсетілген.
            </p>
            <Link to="/read" className="content__link">
              Электронды оқулық
            </Link>
          </div>
          <img src="book_img.png" alt="" className="content__img" />
        </div>
        <div className="annotation">
          <h2 className="annotation__title">Электронды оқулықтың аннотациясы</h2>
          <p className="annotation__descr">
            Машиналық оқыту (Machine Learning) - оқуға қабілетті алгоритмдерді құру әдістерін
            зерттейтін жасанды интеллекттің кең бөлімі. Оқытудың екі түрі бар. Прецеденттер бойынша
            оқыту немесе индуктивті оқыту жеке эмпирикалық мәліметтер бойынша жалпы заңдылықтарды
            анықтауға негізделген.Машиналық оқытудың негзігі ұғымдарымен, машиналық оқытудың негізгі
            алгоритмдерімен оларды қолдану ерекшеліктерімен танысу. Оқулықтың мақсаты машиналық
            оқыту есептерін шешу әдістеріне және курс мәселелеріне кіріспе. Ол инженерлік білім
            әдістерін және мазмұнын, Машиналық оқыту жүйесі ретінде нейрон желілерін және эксперттік
            жүйелердің орны мен ерекшеліктерін оқытады.
          </p>
        </div>
        <div className="avtor">
          <h2 className="avtor__title">Автор туралы</h2>
          <div className="texts">
            <ul className="texts__menu">
              <li className="texts__item">
                <h3 className="texts__title">
                  <b>Аты-жөні: </b> АБИБУЛЛАЕВА АЙМАН АБИБУЛЛАҚЫЗЫ
                </h3>
                <p className="texts__descr">
                  <b>Құрылымы</b> (бөлім, факультет, кафедра): Инженерия факультеті, Компьютерлік
                  инженерия кафедрасы{' '}
                </p>
              </li>
              <li className="texts__item">
                <p className="texts__descr">
                  <b>Қызметі</b> (ғылыми дәрежесі, атағы): Ғылыми дәрежесі, атағы: PhD, аға оқытушы
                </p>
              </li>
              <li className="texts__item">
                <p className="texts__descr">
                  <b>Оқыту курстары:</b>
                </p>
                <ol className="lists">
                  <li>Жасанды интеллект</li>
                  <li>Машиналық оқыту</li>
                  <li>Терең оқыту</li>
                  <li>Роботтық жүйелер</li>
                  <li>Бұлтты технологиялар</li>
                </ol>
              </li>
              <li className="texts__item">
                <p className="texts__descr">
                  <b>Кампус: </b> Кентау қаласы, Кентау қаласы, №1 ғимарат, №306{' '}
                </p>
              </li>
              <li className="texts__item">
                <p className="texts__descr">
                  <b>Телефон:</b> +7 (702) 0926891
                </p>
              </li>
              <li className="texts__item">
                <p className="texts__descr">
                  <b>Email:</b> aiman.abibullayeva@ayu.edu.kz
                </p>
              </li>
            </ul>
            <img className="texts__img" src="photo.jpeg" alt="" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Content;
