// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MaduraiLandRegistry {
    struct Land {
        address owner;
        string cid;
        bytes32 sha256Digest;
        bytes32 sha3Digest;
        uint256 areaCents;
        uint256 priceInr;
        bool verified;
        bool exists;
    }

    address public immutable admin;
    mapping(address => bool) public registrars;
    mapping(uint256 => Land) private lands;
    mapping(uint256 => address[]) private history;

    event RegistrarSet(address indexed account, bool enabled);
    event LandRegistered(uint256 indexed landId, address indexed owner, string cid, bytes32 sha256Digest, bytes32 sha3Digest);
    event OwnershipTransferred(uint256 indexed landId, address indexed from, address indexed to);
    event LandPriceUpdated(uint256 indexed landId, uint256 oldPrice, uint256 newPrice);
    event LandAreaUpdated(uint256 indexed landId, uint256 oldArea, uint256 newArea);
    event LandVerified(uint256 indexed landId, address indexed by);

    modifier onlyAdmin() { require(msg.sender == admin, "not admin"); _; }
    modifier onlyRegistrar() { require(registrars[msg.sender], "not registrar"); _; }
    modifier landExists(uint256 landId) { require(lands[landId].exists, "unknown land"); _; }

    constructor() {
        admin = msg.sender;
        registrars[msg.sender] = true;
        emit RegistrarSet(msg.sender, true);
    }

    function setRegistrar(address account, bool enabled) external onlyAdmin {
        require(account != address(0), "zero address");
        registrars[account] = enabled;
        emit RegistrarSet(account, enabled);
    }

    function registerLand(uint256 landId, address owner, string calldata cid, bytes32 sha256Digest,
                          bytes32 sha3Digest, uint256 areaCents, uint256 priceInr) external onlyRegistrar {
        require(!lands[landId].exists, "already registered");
        require(owner != address(0), "zero address");
        require(bytes(cid).length > 0, "empty cid");
        lands[landId] = Land(owner, cid, sha256Digest, sha3Digest, areaCents, priceInr, false, true);
        history[landId].push(owner);
        emit LandRegistered(landId, owner, cid, sha256Digest, sha3Digest);
    }

    function transferOwnership(uint256 landId, address newOwner) external onlyRegistrar landExists(landId) {
        require(newOwner != address(0), "zero address");
        address previous = lands[landId].owner;
        lands[landId].owner = newOwner;
        history[landId].push(newOwner);
        emit OwnershipTransferred(landId, previous, newOwner);
    }

    function updateLandPrice(uint256 landId, uint256 priceInr) external onlyRegistrar landExists(landId) {
        uint256 old = lands[landId].priceInr;
        lands[landId].priceInr = priceInr;
        emit LandPriceUpdated(landId, old, priceInr);
    }

    function updateLandArea(uint256 landId, uint256 areaCents) external onlyRegistrar landExists(landId) {
        uint256 old = lands[landId].areaCents;
        lands[landId].areaCents = areaCents;
        emit LandAreaUpdated(landId, old, areaCents);
    }

    function verifyLand(uint256 landId) external onlyRegistrar landExists(landId) {
        lands[landId].verified = true;
        emit LandVerified(landId, msg.sender);
    }

    function getLand(uint256 landId) external view landExists(landId) returns (Land memory) {
        return lands[landId];
    }

    function getHistory(uint256 landId) external view landExists(landId) returns (address[] memory) {
        return history[landId];
    }

    function verifyRecord(uint256 landId, bytes32 sha256Digest, bytes32 sha3Digest)
        external view landExists(landId) returns (bool) {
        Land storage l = lands[landId];
        return l.sha256Digest == sha256Digest && l.sha3Digest == sha3Digest;
    }
}
