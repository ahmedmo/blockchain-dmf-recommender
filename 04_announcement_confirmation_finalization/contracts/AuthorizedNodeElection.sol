// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
    ========================================================================
    Contract Name: AuthorizedNodeElection
    ========================================================================

    الفكرة العامة:
    هذا العقد لا يحسب A_k.
    اختيار A_k يتم خارج السلسلة بواسطة Python.

    وظيفة العقد:
    1) تحميل أعضاء المجتمع.
    2) استقبال إعلان A_k.
    3) استقبال تأكيدات أعضاء المجتمع.
    4) تثبيت A_k عند الوصول إلى الأغلبية البسيطة.
    5) إلغاء الإعلان إذا انتهت مهلة التأكيد دون الوصول إلى الأغلبية.

    التصميم المعتمد:
    - إعلان واحد معلّق فقط لكل مجتمع.
    - لا يوجد snapshot.
    - A_k هو user_id وليس wallet address.
    - A_k لا يشترط أن يكون عضوًا في المجتمع.
    - المعلن والمؤكدون يجب أن يكونوا أعضاء في المجتمع.
*/

contract AuthorizedNodeElection {

    /*
        owner:
        هو الحساب الذي ينشر العقد.
        هذا الحساب فقط يستطيع تحميل أعضاء المجتمع.
    */
    address public owner;

    /*
        confirmationWindowSeconds:
        مدة السماح بالتأكيدات بعد إعلان A_k.

        مثال:
        300  = خمس دقائق
        3600 = ساعة واحدة

        هذه القيمة تُحدد عند نشر العقد من Remix.
    */
    uint256 public confirmationWindowSeconds;

    /*
        Announcement:
        يمثل حالة إعلان A_k لمجتمع معين.

        Ak:
        رقم المستخدم المختار كعقدة موثوقة ومخوّلة.

        resultHash:
        بصمة نتيجة الاختيار خارج السلسلة.

        confirmCount:
        عدد التأكيدات الصحيحة المقبولة.

        exists:
        هل يوجد إعلان معلّق؟

        finalized:
        هل تم تثبيت A_k نهائيًا؟

        announcer:
        عنوان العضو الذي أعلن A_k.

        announcedAt:
        وقت إنشاء الإعلان.

        deadline:
        آخر وقت مسموح به لقبول التأكيدات.
    */
    struct Announcement {
        uint256 Ak;
        bytes32 resultHash;
        uint256 confirmCount;
        bool exists;
        bool finalized;
        address announcer;
        uint256 announcedAt;
        uint256 deadline;
    }

    /*
        isMember[communityId][address] = true

        هذا يعني أن address عضو في المجتمع communityId.
    */
    mapping(uint256 => mapping(address => bool)) public isMember;

    /*
        communitySize[communityId]
        يخزن عدد أعضاء المجتمع.
    */
    mapping(uint256 => uint256) public communitySize;

    /*
        announcements[communityId]
        يخزن إعلان A_k للمجتمع.

        لا يوجد snapshot.
        لكل مجتمع إعلان واحد فقط في كل مرة.
    */
    mapping(uint256 => Announcement) public announcements;

    /*
        hasConfirmed[communityId][address]
        يمنع العضو من التأكيد أكثر من مرة.
    */
    mapping(uint256 => mapping(address => bool)) public hasConfirmed;

    /*
        confirmersList:
        نخزن المؤكدين حتى نستطيع تنظيف hasConfirmed
        عند إلغاء إعلان منتهي أو عند reset يدوي.
    */
    mapping(uint256 => address[]) private confirmersList;

    /*
        finalizedAk[communityId]
        يخزن القائد النهائي بعد التثبيت.
    */
    mapping(uint256 => uint256) public finalizedAk;

    /*
        Events:
        تستخدمها Python أو أي خدمة خارجية لمعرفة ما حدث داخل العقد.
    */
    event CommunityMembersLoaded(
        uint256 indexed communityId,
        uint256 loadedCount,
        uint256 totalCommunitySize
    );

    event AkAnnounced(
        uint256 indexed communityId,
        uint256 Ak,
        bytes32 resultHash,
        address indexed announcer,
        uint256 announcedAt,
        uint256 deadline
    );

    event AkConfirmed(
        uint256 indexed communityId,
        uint256 Ak,
        address indexed confirmer,
        uint256 confirmCount,
        uint256 requiredConfirmations
    );

    event AkFinalized(
        uint256 indexed communityId,
        uint256 Ak,
        bytes32 resultHash
    );

    event PendingAnnouncementExpired(
        uint256 indexed communityId,
        uint256 oldAk,
        bytes32 oldResultHash
    );

    event PendingAnnouncementReset(
        uint256 indexed communityId,
        uint256 oldAk,
        bytes32 oldResultHash
    );

    /*
        فقط owner يستطيع تنفيذ بعض الدوال الإدارية.
    */
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /*
        فقط عضو المجتمع يستطيع الإعلان أو التأكيد.
    */
    modifier onlyCommunityMember(uint256 communityId) {
        require(
            isMember[communityId][msg.sender],
            "Caller is not a community member"
        );
        _;
    }

    /*
        constructor:
        يتم تنفيذه عند نشر العقد.

        _confirmationWindowSeconds:
        مدة التأكيد بالثواني.

        مثال للنشر من Remix:
        300  = خمس دقائق
        3600 = ساعة واحدة
    */
    constructor(uint256 _confirmationWindowSeconds) {
        require(
            _confirmationWindowSeconds > 0,
            "Confirmation window must be greater than zero"
        );

        owner = msg.sender;
        confirmationWindowSeconds = _confirmationWindowSeconds;
    }

    /*
        ====================================================================
        loadCommunityMembers
        ====================================================================

        الهدف:
        تحميل أعضاء مجتمع معين إلى العقد.

        ملاحظة:
        نحمّل wallet addresses وليس user_id.
    */
    function loadCommunityMembers(
        uint256 communityId,
        address[] calldata members
    ) external onlyOwner {
        require(members.length > 0, "Empty member list");

        uint256 added = 0;

        for (uint256 i = 0; i < members.length; i++) {
            address memberAddress = members[i];

            require(memberAddress != address(0), "Invalid member address");

            if (!isMember[communityId][memberAddress]) {
                isMember[communityId][memberAddress] = true;
                communitySize[communityId] += 1;
                added += 1;
            }
        }

        emit CommunityMembersLoaded(
            communityId,
            added,
            communitySize[communityId]
        );
    }

    /*
        ====================================================================
        requiredConfirmations
        ====================================================================

        الأغلبية البسيطة:
        floor(communitySize / 2) + 1

        مثال:
        104 عضو → 53 تأكيد.
    */
    function requiredConfirmations(
        uint256 communityId
    ) public view returns (uint256) {
        uint256 n = communitySize[communityId];

        require(n > 0, "Unknown or empty community");

        return (n / 2) + 1;
    }

    /*
        ====================================================================
        announceAk
        ====================================================================

        الهدف:
        إعلان A_k لمجتمع معين.

        الشروط:
        - المعلن يجب أن يكون عضوًا في المجتمع.
        - لا يوجد إعلان معلّق حاليًا.
        - لا يوجد A_k مثبت سابقًا.
        - resultHash يجب ألا يكون صفرًا.

        مهم:
        لا نتحقق أن A_k عضو في المجتمع.
    */
    function announceAk(
        uint256 communityId,
        uint256 Ak,
        bytes32 resultHash
    ) external onlyCommunityMember(communityId) {
        require(resultHash != bytes32(0), "Invalid result hash");

        Announcement storage a = announcements[communityId];

        require(
            !a.finalized,
            "A_k already finalized for this community"
        );

        require(
            !a.exists,
            "Pending announcement already exists"
        );

        uint256 currentTime = block.timestamp;
        uint256 deadline = currentTime + confirmationWindowSeconds;

        a.Ak = Ak;
        a.resultHash = resultHash;
        a.confirmCount = 1;
        a.exists = true;
        a.finalized = false;
        a.announcer = msg.sender;
        a.announcedAt = currentTime;
        a.deadline = deadline;

        /*
            المعلن يُحتسب كتأكيد أول.
        */
        hasConfirmed[communityId][msg.sender] = true;
        confirmersList[communityId].push(msg.sender);

        emit AkAnnounced(
            communityId,
            Ak,
            resultHash,
            msg.sender,
            currentTime,
            deadline
        );

        emit AkConfirmed(
            communityId,
            Ak,
            msg.sender,
            a.confirmCount,
            requiredConfirmations(communityId)
        );

        if (a.confirmCount >= requiredConfirmations(communityId)) {
            _finalizeAk(communityId);
        }
    }

    /*
        ====================================================================
        confirmAk
        ====================================================================

        الهدف:
        تأكيد الإعلان الموجود.

        الشروط:
        - المؤكد يجب أن يكون عضوًا في المجتمع.
        - الإعلان يجب أن يكون موجودًا.
        - الإعلان لم يتم تثبيته بعد.
        - لم تنتهِ مهلة التأكيد.
        - قيمة Ak يجب أن تطابق الإعلان.
        - العضو لم يؤكد سابقًا.
    */
    function confirmAk(
        uint256 communityId,
        uint256 Ak
    ) external onlyCommunityMember(communityId) {
        Announcement storage a = announcements[communityId];

        require(a.exists, "No pending announcement");

        require(!a.finalized, "A_k already finalized");

        require(
            block.timestamp <= a.deadline,
            "Confirmation deadline passed"
        );

        require(a.Ak == Ak, "A_k mismatch");

        require(
            !hasConfirmed[communityId][msg.sender],
            "Already confirmed"
        );

        hasConfirmed[communityId][msg.sender] = true;
        confirmersList[communityId].push(msg.sender);

        a.confirmCount += 1;

        emit AkConfirmed(
            communityId,
            Ak,
            msg.sender,
            a.confirmCount,
            requiredConfirmations(communityId)
        );

        if (a.confirmCount >= requiredConfirmations(communityId)) {
            _finalizeAk(communityId);
        }
    }

    /*
        ====================================================================
        _finalizeAk
        ====================================================================

        الهدف:
        تثبيت A_k نهائيًا بعد تحقق الأغلبية.
    */
    function _finalizeAk(uint256 communityId) internal {
        Announcement storage a = announcements[communityId];

        require(a.exists, "No announcement to finalize");

        require(!a.finalized, "Already finalized");

        require(
            a.confirmCount >= requiredConfirmations(communityId),
            "Threshold not reached"
        );

        a.finalized = true;
        finalizedAk[communityId] = a.Ak;

        emit AkFinalized(
            communityId,
            a.Ak,
            a.resultHash
        );
    }

    /*
        ====================================================================
        isAnnouncementExpired
        ====================================================================

        الهدف:
        معرفة هل الإعلان المعلق انتهت مهلته أم لا.

        يرجع true فقط إذا:
        - يوجد إعلان.
        - لم يتم تثبيته.
        - الوقت الحالي تجاوز deadline.
    */
    function isAnnouncementExpired(
        uint256 communityId
    ) public view returns (bool) {
        Announcement storage a = announcements[communityId];

        if (!a.exists) {
            return false;
        }

        if (a.finalized) {
            return false;
        }

        return block.timestamp > a.deadline;
    }

    /*
        ====================================================================
        expirePendingAnnouncement
        ====================================================================

        الهدف:
        إلغاء إعلان انتهت مهلته ولم يصل إلى الأغلبية.

        من يستطيع استدعاءها؟
        - owner
        - أو أي عضو من المجتمع

        لماذا نسمح للعضو؟
        حتى لا يبقى إعلان خبيث أو خاطئ معلقًا إذا انتهت مهلته.

        ملاحظة:
        الإلغاء لا يحدث تلقائيًا.
        يجب إرسال transaction لاستدعاء هذه الدالة بعد انتهاء الوقت.
    */
    function expirePendingAnnouncement(
        uint256 communityId
    ) external {
        require(
            msg.sender == owner || isMember[communityId][msg.sender],
            "Caller must be owner or community member"
        );

        Announcement storage a = announcements[communityId];

        require(a.exists, "No pending announcement");

        require(!a.finalized, "Cannot expire finalized A_k");

        require(
            block.timestamp > a.deadline,
            "Confirmation deadline has not passed"
        );

        uint256 oldAk = a.Ak;
        bytes32 oldResultHash = a.resultHash;

        _clearConfirmationFlags(communityId);

        delete announcements[communityId];

        emit PendingAnnouncementExpired(
            communityId,
            oldAk,
            oldResultHash
        );
    }

    /*
        ====================================================================
        resetPendingAnnouncement
        ====================================================================

        الهدف:
        إلغاء يدوي لإعلان معلّق قبل انتهاء الوقت.

        من يستدعيها؟
        owner فقط.

        تستخدم أثناء التجارب إذا تم إدخال إعلان خاطئ.
        لا تعمل بعد finalization.
    */
    function resetPendingAnnouncement(
        uint256 communityId
    ) external onlyOwner {
        Announcement storage a = announcements[communityId];

        require(a.exists, "No pending announcement");

        require(
            !a.finalized,
            "Cannot reset finalized A_k"
        );

        uint256 oldAk = a.Ak;
        bytes32 oldResultHash = a.resultHash;

        _clearConfirmationFlags(communityId);

        delete announcements[communityId];

        emit PendingAnnouncementReset(
            communityId,
            oldAk,
            oldResultHash
        );
    }

    /*
        ====================================================================
        _clearConfirmationFlags
        ====================================================================

        الهدف:
        تنظيف hasConfirmed عند إلغاء إعلان.

        السبب:
        إذا تم إلغاء إعلان، يجب أن يستطيع الأعضاء التأكيد مرة أخرى
        عند ظهور إعلان جديد.
    */
    function _clearConfirmationFlags(uint256 communityId) internal {
        address[] storage voters = confirmersList[communityId];

        for (uint256 i = 0; i < voters.length; i++) {
            hasConfirmed[communityId][voters[i]] = false;
        }

        delete confirmersList[communityId];
    }

    /*
        ====================================================================
        getAnnouncement
        ====================================================================

        الهدف:
        قراءة الحالة الأساسية لإعلان مجتمع معين.

        تم تقليل عدد القيم الراجعة لتجنب خطأ Stack too deep.
    */
    function getAnnouncement(
        uint256 communityId
    )
        external
        view
        returns (
            uint256 Ak,
            bytes32 resultHash,
            uint256 confirmCount,
            bool exists,
            bool finalized,
            address announcer
        )
    {
        Announcement storage a = announcements[communityId];

        return (
            a.Ak,
            a.resultHash,
            a.confirmCount,
            a.exists,
            a.finalized,
            a.announcer
        );
    }

    /*
        ====================================================================
        getAnnouncementTiming
        ====================================================================

        الهدف:
        قراءة وقت الإعلان، وقت انتهاء المهلة، وهل الإعلان منتهي أم لا.
    */
    function getAnnouncementTiming(
        uint256 communityId
    )
        external
        view
        returns (
            uint256 announcedAt,
            uint256 deadline,
            bool expired
        )
    {
        Announcement storage a = announcements[communityId];

        return (
            a.announcedAt,
            a.deadline,
            isAnnouncementExpired(communityId)
        );
    }

    /*
        ====================================================================
        getCommunityConfirmationInfo
        ====================================================================

        الهدف:
        قراءة حجم المجتمع، عدد التأكيدات المطلوبة، وعدد التأكيدات الحالية.
    */
    function getCommunityConfirmationInfo(
        uint256 communityId
    )
        external
        view
        returns (
            uint256 size,
            uint256 requiredCount,
            uint256 currentConfirmCount
        )
    {
        Announcement storage a = announcements[communityId];

        uint256 req = 0;

        if (communitySize[communityId] > 0) {
            req = requiredConfirmations(communityId);
        }

        return (
            communitySize[communityId],
            req,
            a.confirmCount
        );
    }

    /*
        ====================================================================
        hasAddressConfirmed
        ====================================================================

        الهدف:
        معرفة هل عنوان معين أكد إعلان المجتمع أم لا.
    */
    function hasAddressConfirmed(
        uint256 communityId,
        address account
    ) external view returns (bool) {
        return hasConfirmed[communityId][account];
    }

    /*
        ====================================================================
        getConfirmersCount
        ====================================================================

        الهدف:
        قراءة عدد العناوين التي أكدت.
    */
    function getConfirmersCount(
        uint256 communityId
    ) external view returns (uint256) {
        return confirmersList[communityId].length;
    }
}