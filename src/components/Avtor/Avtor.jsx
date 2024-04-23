import React from 'react';

import './avtor.scss';

function Avtor() {
  return (
    <div className="container">
      <div className="about">
        <div className="about__titles">
          <img className="about__img" src="photo.jpeg" alt="" />
          <h2 className="about__title">Абибуллаева Айман Абибуллақызы</h2>
        </div>
        <div className="about__inner">
          <ul className="about__menu">
            <li className="about__item">
              <h3 className="about__item-title">БІЛІМІ</h3>
              <p className="about__descr">
                <li> №6 М.Мәметова атындағы орта мектеп</li>
                <li>
                  Бакалавр: Қожа Ахмет Ясауи атындағы халықаралық қазақ-түрік университеті,
                  мамандығы: Ақпараттық жүйелер.
                </li>
                <li>
                  Магистратура: Қожа Ахмет Ясауи атындағы халықаралық қазақ-түрік университеті,
                  мамандығы: Математикалық және компьютерлік модельдеу.
                </li>
                <li>
                  Докторантура: Гази университеті (Анкара) , мамандығы: Компьютерлік инженерия.
                </li>

                <li>Ғылыми дәрежесі, атағы: PhD, аға оқытушы</li>
              </p>
            </li>
            <li className="about__item">
              <h3 className="about__item-title">ҒЫЛЫМИ ДӘРЕЖЕСІ, АТАҒЫ</h3>
              <p className="about__descr">PhD, аға оқытушы</p>
            </li>
            <li className="about__item">
              <h3 className="about__item-title">МАҚСАТЫ</h3>
              <p className="about__descr">
                Перспективті жоғары оқу орнында жұмыс жасап, Қазақстанның өркендеуіне және
                IT-индустриясын дамытатын студенттерді дайындауда өз үлесімді қосу
              </p>
            </li>
            <li className="about__item">
              <h3 className="about__item-title">ЕРЕКШЕ ҚАСИЕТІ</h3>
              <p className="about__descr">
                Қарапайымдылық, көппен тіл таба алу, коллективті басқара алу, IT саласында үнемі
                іздену
              </p>
            </li>
          </ul>
          <ul className="about__menu">
            <li className="about__item">
              <h3 className="about__item-title">ЖҰМЫС ТƏЖІРИБЕСІ</h3>
              <p className="about__descr">
                <li>
                  2023-қазіргі уақытта - Қожа Ахмет Ясауи атындағы халықаралық қазақ-түрік
                  университеті, “Компьютерлік инженерия” кафедрасында PhD, аға оқытушы.
                </li>
              </p>
            </li>
            <li className="about__item">
              <h3 className="about__item-title">ҒЫЛЫМИ ЕҢБЕКТЕРІ</h3>
              <p className="about__descr">
                Ғылыми мақалалары халықаралық, республикалық конференция жинақтарында, Отандық және
                халықаралық ғылыми журналдарда, Қазақстан Республикасы Білім және ғылым
                министрлігінің Білім және ғылым саласындағы бақылау комитетімен (ҚР БҒМ ККСОН)
                ұсынылған басылымдарында, Scopus-тың индекстелген импакт-факторлы журналдарында
                жарияланды.
              </p>
            </li>
            <li className="about__item">
              <h3 className="about__item-title">ҚОСЫМША АҚПАРАТ</h3>
              <p className="about__descr">
                «Ақпараттық-коммуникациялық технологиялар» бағыты бойынша Республикалық
                оқу-әдістемелік бірлестікке (УМО РУМС)
              </p>
            </li>
          </ul>
        </div>
        <div className="about__icons">
          <li>
            <img src="img/icon2.png" alt="" />
            <p>aiman.abibullayeva@ayu.edu.kz</p>
          </li>
          <li>
            <img src="img/icon1.png" alt="" />
            <p>+7 (702) 0926891</p>
          </li>
        </div>
      </div>
    </div>
  );
}

export default Avtor;
