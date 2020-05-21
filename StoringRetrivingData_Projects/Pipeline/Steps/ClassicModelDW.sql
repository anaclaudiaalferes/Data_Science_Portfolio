DROP DATABASE IF EXISTS classicmodelsDW;
CREATE DATABASE classicmodelsDW;

USE classicmodelsDW;

Create table `dim_products`(
  `productID` varchar(15),
  `productName` varchar(70),
  `productLine` varchar(50),
  `productScale` varchar(10),
  `productVendor` varchar(50),
  `quantityInStock` smallint(6),
  `buyPrice` decimal(10,2),
  `MSRP` decimal(10,2),
  PRIMARY KEY (`productID`)
);

Create table `dim_customer`(
  `customerID` int(11),
  `customerName` varchar(50),
  `city` varchar(50),
  `state` varchar(50),
  `country` varchar(50),
  `creditLimit` decimal(10,2),
  PRIMARY KEY (`customerID`)
);

CREATE TABLE `dim_employees` (
  `employeeID` int(11),
  `Name` varchar(50),
  `jobTitle` varchar(50),
  PRIMARY KEY (`employeeID`)
);

CREATE TABLE `dim_offices` (
  `officeID` varchar(10),
  `city` varchar(50),
  `state` varchar(50),
  `country` varchar(50),
  `territory` varchar(10),
  PRIMARY KEY (`officeID`)
);

CREATE TABLE `fact_orders` (
  `orderID` int(11),
  `productID` varchar(15),
  `orderDate` date,
  `shippedDate` date,
  `status` varchar(15),
  `customerID` int(11),
  `quantity` int(11),
  `priceEach` decimal(10,2),
  `employeeID` int(11),
  `officeID` varchar(10),
  PRIMARY KEY (`orderID`,`productID`),
  FOREIGN KEY (`productID`) REFERENCES `dim_products` (`productID`),
  FOREIGN KEY (`customerID`) REFERENCES `dim_customer` (`customerID`),
  FOREIGN KEY (`employeeID`) REFERENCES `dim_employees` (`employeeID`),
  FOREIGN KEY (`officeID`) REFERENCES `dim_offices` (`officeID`)
)
