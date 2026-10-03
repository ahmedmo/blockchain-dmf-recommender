
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
    RecommendationDelivery

    This contract supports the recommendation delivery stage.

    It does not store full recommendations on-chain.
    Instead, it stores only the hash of each recommendation payload.

    Main functions:
    - Register the trusted authorized node A_k for each community.
    - Register community members.
    - Allow only A_k to store recommendation hashes.
    - Reject hashes for users who are not registered community members.
    - Verify whether an off-chain payload matches the on-chain hash.
*/

contract RecommendationDelivery {

    address public owner;

    struct RecommendationRecord {
        bytes32 recommendationHash;
        address issuer;
        uint256 timestamp;
        bool exists;
    }

    // communityId => authorized node A_k address
    mapping(uint256 => address) public authorizedNode;

    // communityId => userAddress => membership status
    mapping(uint256 => mapping(address => bool)) public isCommunityMember;

    // communityId => number of registered members
    mapping(uint256 => uint256) public communityMemberCount;

    /*
        records[communityId][user][version][topN] = RecommendationRecord
    */
    mapping(uint256 => mapping(address => mapping(uint256 => mapping(uint256 => RecommendationRecord)))) private records;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }

    modifier onlyAuthorizedNode(uint256 communityId) {
        require(
            msg.sender == authorizedNode[communityId],
            "Only authorized node can store recommendation hashes"
        );
        _;
    }

    /*
        Set the trusted authorized node A_k for a community.
        This is executed by the owner after reading the finalized announcement results.
    */
    function setAuthorizedNode(
        uint256 communityId,
        address node
    ) external onlyOwner {
        require(node != address(0), "Invalid authorized node address");
        authorizedNode[communityId] = node;
    }

    /*
        Register a batch of community members.
        The contract uses user addresses, not raw dataset user IDs.
    */
    function registerCommunityMembers(
        uint256 communityId,
        address[] calldata members
    ) external onlyOwner {
        for (uint256 i = 0; i < members.length; i++) {
            address member = members[i];
            require(member != address(0), "Invalid member address");

            if (!isCommunityMember[communityId][member]) {
                isCommunityMember[communityId][member] = true;
                communityMemberCount[communityId] += 1;
            }
        }
    }

    /*
        Check whether an address is registered as a member of a community.
    */
    function checkCommunityMember(
        uint256 communityId,
        address member
    ) external view returns (bool) {
        return isCommunityMember[communityId][member];
    }

    /*
        Store the recommendation hash only.

        Acceptance conditions:
        1. msg.sender must be A_k of the corresponding community.
        2. target user must be a registered member of that community.
        3. the same key must not already exist:
           communityId + user + version + topN.
    */
    function storeRecommendationHash(
        uint256 communityId,
        address user,
        uint256 version,
        uint256 topN,
        bytes32 recommendationHash
    ) external onlyAuthorizedNode(communityId) {
        require(user != address(0), "Invalid user address");
        require(topN > 0, "Invalid topN value");
        require(recommendationHash != bytes32(0), "Invalid recommendation hash");

        require(
            isCommunityMember[communityId][user],
            "Target user is not a registered member of this community"
        );

        require(
            records[communityId][user][version][topN].exists == false,
            "Recommendation hash already exists for this key"
        );

        records[communityId][user][version][topN] = RecommendationRecord({
            recommendationHash: recommendationHash,
            issuer: msg.sender,
            timestamp: block.timestamp,
            exists: true
        });
    }

    /*
        Check whether a recommendation record already exists.
    */
    function recordExists(
        uint256 communityId,
        address user,
        uint256 version,
        uint256 topN
    ) external view returns (bool) {
        return records[communityId][user][version][topN].exists;
    }

    /*
        Get the stored recommendation hash.
    */
    function getRecommendationHash(
        uint256 communityId,
        address user,
        uint256 version,
        uint256 topN
    ) external view returns (bytes32) {
        require(
            records[communityId][user][version][topN].exists,
            "Recommendation record not found"
        );

        return records[communityId][user][version][topN].recommendationHash;
    }

    /*
        Verify whether the locally computed hash matches the stored on-chain hash.

        This verifies integrity only.
        It does not evaluate recommendation quality.
    */
    function verifyRecommendationHash(
        uint256 communityId,
        address user,
        uint256 version,
        uint256 topN,
        bytes32 receivedHash
    ) external view returns (bool) {
        if (!records[communityId][user][version][topN].exists) {
            return false;
        }

        return records[communityId][user][version][topN].recommendationHash == receivedHash;
    }

    /*
        Get the full on-chain record metadata for audit and validation.
    */
    function getRecommendationRecord(
        uint256 communityId,
        address user,
        uint256 version,
        uint256 topN
    ) external view returns (
        bytes32 recommendationHash,
        address issuer,
        uint256 timestamp,
        bool exists
    ) {
        RecommendationRecord memory record = records[communityId][user][version][topN];

        return (
            record.recommendationHash,
            record.issuer,
            record.timestamp,
            record.exists
        );
    }
}
