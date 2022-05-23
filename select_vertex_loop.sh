#!/bin/bash

# for i in 1825 1834 1846 1596 1597 1598 1599 1600 1606 1608 1609 1614 1621 1623 1624 1625 1626 1627 1628 1629 1630 1631 1632 1633 1634 1635 1636 1637 1638 1639 1640 1641 1642 1643 1644 1645 1646 1647 1648 1649 1650 1651 1652 1653 1654 1655 1656 1657 1658 1659 1660 1661 1662 1663 1664 1665 1666 1667 1668 1669 1671 1672 1673 1674 1675 1676 1677 1678 1679 1680 1681 1682 1683 1684 1685 1686 1687 
for i in 1689 1692 1693 1694 1695 1711 1716 1717 1718 1719 1722 1724 1725 1726 1727 1728 1729 1732 1749 1750 1759 1760 1761 1762 1763 1766 1767 1768 1769 1770 1784 1785 1787 1788 1789 1790 1795 1796 1797 1798 1799 1807 1809 1810 1811 1812 1813 1814 1815 1817 1818 1819 1821 1826 1828 1829 1830 1831 1832 1833 1835 1836 1837 1838 1839 1843 1849 1850 1851 1852 1853 1854 1859 1864 1865 1867 1868 1870 1872 1873 1874 1875 1876 1877 1878 1879 1880 1891 1909 1910 1911 1912 1913 1914 1915 1916 1917 1918 1919 1920 1921 1922 1923 1924 1925 1926 1928 1930 1932 1933 1934 1935 1936 1937 1939 1940 1941 1942 1943 1944 1945 1948 1949 1950 1952 1500 1514 1538 1539 1540
do
    for j in 1.4  ### vertex X^2+Y^2
    do
        
        for m in 1.0 ### vertex Z
        do
            echo ${i}
            echo ${j}

            echo ${m}
#             ##################### min value 없음
#             python select_vertex_outAC.py \
#             -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_h5/ \
#             -o /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_Rho_${j}_ZL_${m}/  \
#             --vtxRho ${j} --vtxz ${m} --type 0 --min 1 --minvalue 400 --dvertex 1 --cut 1

#             echo "FN clear"

#             python select_vertex_outAC.py \
#             -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_h5/ \
#             -o /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_Rho_${j}_ZL_${m}/ \
#             --vtxRho ${j} --vtxz ${m} --type 1 --min 1 --minvalue 400 --dvertex 1 --cut 1

#             echo "ME clear"

            python data_subrun_divide.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_Rho_${j}_ZL_${m}/ \
            --evenoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_even_Rho_${j}_ZL_${m}_noDIN/ \
            --oddoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/FN_cut_odd_Rho_${j}_ZL_${m}_noDIN/

            echo "FN div clear"

            python data_subrun_divide.py \
            -i /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_Rho_${j}_ZL_${m}/  \
            --evenoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_even_Rho_${j}_ZL_${m}_noDIN/ \
            --oddoutput /store/hep/users/yewzzang/JSNS2/com_data/r00${i}/ME_cut_odd_Rho_${j}_ZL_${m}_noDIN/

            echo "ME div clear"
    
        done
    done
done



