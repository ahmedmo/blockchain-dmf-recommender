
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
    RecommendationDelivery

    الهدف العام:
    - لا يخزن التوصيات الكاملة على البلوكشين.
    - يخزن فقط hash يمثل حزمة توصيات المستخدم.
    - يسجل العقدة الموثوقة A_k لكل مجتمع.
    - يسجل أعضاء كل مجتمع حتى لا يتم تخزين Hash لتوصيات مستخدم غير عضو.
    - يسمح فقط لـ A_k الخاصة بالمجتمع بتخزين Hash التوصيات.

    مفتاح التخزين:
    communityId + userAddress + version + topN

    ملاحظة منهجية:
    التوصيات الفعلية تبقى خارج البلوكشين في Payload JSON
    أو في قناة off-chain مثل API أو node-to-node communication.
*/

contract RecommendationDelivery {

    address public owner;

    struct RecommendationRecord {
        bytes32 recommendationHash;   // Hash of recommendation payload
        address issuer;               // Authorized node A_k that stored the hash
        uint256 timestamp;            // Block timestamp
        bool exists;                  // Record existence flag
    }

    // communityId => authorized node A_k
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
        تعيين العقدة الموثوقة A_k لمجتمع معين.
        هذا يتم من قبل owner بعد قراءة نتائج التصويت.
    */
    function setAuthorizedNode(
        uint256 communityId,
        address node
    ) external onlyOwner {
        require(node != address(0), "Invalid authorized node address");
        authorizedNode[communityId] = node;
    }

    /*
        تسجيل عضو واحد داخل مجتمع.
        العضوية هنا تستخدم userAddress وليس user_id.
    */
    function registerCommunityMember(
        uint256 communityId,
        address member
    ) external onlyOwner {
        require(member != address(0), "Invalid member address");

        if (!isCommunityMember[communityId][member]) {
            isCommunityMember[communityId][member] = true;
            communityMemberCount[communityId] += 1;
        }
    }

    /*
        تسجيل مجموعة أعضاء داخل مجتمع.
        هذا يقلل عدد المعاملات مقارنة بتسجيل كل عضو منفردًا.
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
        فحص هل النود عضو في المجتمع.
    */
    function checkCommunityMember(
        uint256 communityId,
        address member
    ) external view returns (bool) {
        return isCommunityMember[communityId][member];
    }

    /*
        تخزين Hash التوصيات فقط.

        شروط القبول:
        1. msg.sender هو A_k لذلك المجتمع.
        2. user عضو مسجل في المجتمع.
        3. السجل غير مخزن سابقًا لنفس communityId + user + version + topN.
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
        فحص وجود سجل سابق.
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
        جلب الهاش المخزون.
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
        مقارنة الهاش المحلي مع الهاش المخزون.
        هذه الدالة تتحقق من سلامة البيانات Integrity فقط.
        لا تتحقق من جودة التوصيات.
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
        جلب تفاصيل السجل لغرض المراجعة والتدقيق.
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
